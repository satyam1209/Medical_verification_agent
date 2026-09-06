"""Fully async ingestion of medical evidence from PubMed, ClinicalTrials.gov, and openFDA.

Run: python ingest.py
"""

import asyncio
import json
import os
import time
import xml.etree.ElementTree as ET
import zlib

import httpx
from dotenv import load_dotenv

load_dotenv()

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, PayloadSchemaType, PointStruct, VectorParams
from fastembed import TextEmbedding

from db_setup import init_db, get_session, now_str
from schema import MedicalRecord

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_DIM = 384
QDRANT_COLLECTION = "medverify_kb"

PUBMED_ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
CLINICALTRIALS_URL = "https://clinicaltrials.gov/api/v2/studies"
OPENFDA_URL = "https://api.fda.gov/drug/event.json"


def _xml_text(element, path, default=""):
    found = element.find(path)
    if found is None:
        return default
    return "".join(found.itertext()).strip()


def _parse_bool(value):
    return bool(int(value)) if value in ("0", "1") else None


def _point_id(record_id, source):
    return zlib.crc32(f"{source}:{record_id}".encode("utf-8"))


# -----------------------------
# 1. PubMed
# -----------------------------
async def fetch_pubmed(client, query, max_results):
    api_key = os.getenv("NCBI_API_KEY")
    email = os.getenv("NCBI_EMAIL", "test@example.com")

    esearch_params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
        "email": email,
    }
    if api_key:
        esearch_params["api_key"] = api_key

    resp = await client.get(PUBMED_ESEARCH_URL, params=esearch_params)
    resp.raise_for_status()
    ids = resp.json().get("esearchresult", {}).get("idlist", [])
    if not ids:
        return []

    efetch_params = {
        "db": "pubmed",
        "id": ",".join(ids),
        "rettype": "abstract",
        "retmode": "xml",
    }
    if api_key:
        efetch_params["api_key"] = api_key

    resp = await client.get(PUBMED_EFETCH_URL, params=efetch_params)
    resp.raise_for_status()
    root = ET.fromstring(resp.content)

    records = []
    for article_el in root.findall(".//PubmedArticle"):
        try:
            pmid = _xml_text(article_el, ".//MedlineCitation/PMID")
            if not pmid:
                continue

            title = _xml_text(article_el, ".//Article/ArticleTitle")
            journal = _xml_text(article_el, ".//Article/Journal/Title")
            pub_date = _xml_text(article_el, ".//Article/Journal/JournalIssue/PubDate/Year")
            if not pub_date:
                pub_date = _xml_text(article_el, ".//Article/Journal/JournalIssue/PubDate/MedlineDate")

            authors = []
            for author_el in article_el.findall(".//Article/AuthorList/Author"):
                last = _xml_text(author_el, "LastName")
                initials = _xml_text(author_el, "Initials")
                if not last:
                    last = _xml_text(author_el, "CollectiveName")
                name = " ".join(p for p in (last, initials) if p)
                if name:
                    authors.append(name)

            abstract_parts = []
            for at_el in article_el.findall(".//Article/Abstract/AbstractText"):
                label = at_el.get("Label")
                text = "".join(at_el.itertext()).strip()
                if label and text:
                    abstract_parts.append(f"{label}: {text}")
                elif text:
                    abstract_parts.append(text)
            abstract = " ".join(abstract_parts)

            if not abstract:
                continue

            metadata = {
                "journal": journal,
                "authors": authors,
                "abstract_first_300": abstract[:300],
            }

            records.append({
                "id": pmid,
                "source": "pubmed",
                "type": "paper",
                "title": title,
                "text": abstract,
                "publish_date": pub_date,
                "last_updated": None,
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                "metadata_json": json.dumps(metadata),
            })
        except Exception as e:
            print(f"WARNING: error parsing pubmed article: {e}")
    return records


# -----------------------------
# 2. ClinicalTrials.gov
# -----------------------------
async def fetch_clinicaltrials(client, query, max_results):
    params = {"query.term": query, "pageSize": max_results}
    resp = await client.get(CLINICALTRIALS_URL, params=params)
    resp.raise_for_status()
    data = resp.json()

    records = []
    for study in data.get("studies", []):
        try:
            proto = study.get("protocolSection", {})
            ident = proto.get("identificationModule", {})
            nct_id = ident.get("nctId", "")
            if not nct_id:
                continue

            title = ident.get("briefTitle", "")
            status = proto.get("statusModule", {}).get("overallStatus", "")
            start_date = proto.get("statusModule", {}).get("startDateStruct", {}).get("date", "")
            last_updated = proto.get("statusModule", {}).get("lastUpdatePostDateStruct", {}).get("date", "")
            conditions = list(proto.get("conditionsModule", {}).get("conditions", []))
            summary = proto.get("descriptionModule", {}).get("briefSummary", "")
            phases = list(proto.get("designModule", {}).get("phases", []))

            text = summary.strip()
            if not text:
                continue

            metadata = {
                "status": status,
                "conditions": conditions,
                "phase": phases,
            }

            records.append({
                "id": nct_id,
                "source": "clinicaltrials",
                "type": "trial",
                "title": title,
                "text": text,
                "publish_date": start_date,
                "last_updated": last_updated,
                "url": f"https://clinicaltrials.gov/study/{nct_id}",
                "metadata_json": json.dumps(metadata),
            })
        except Exception as e:
            print(f"WARNING: error parsing clinicaltrials study: {e}")
    return records


# -----------------------------
# 3. openFDA
# -----------------------------
OPENFDA_RETRIES = 3


async def _openfda_get_with_retry(client, params, attempts=OPENFDA_RETRIES):
    """GET with retry/backoff; 404 is retried in case of transient proxy
    errors, and if it still 404s the caller decides on a fallback query."""
    for attempt in range(1, attempts + 1):
        resp = await client.get(OPENFDA_URL, params=params)
        if resp.status_code != 404 or attempt == attempts:
            return resp
        delay = attempt  # 1s, 2s, ...
        print(f"openFDA: got 404 on attempt {attempt}; retrying in {delay}s")
        await asyncio.sleep(delay)
    return resp


async def fetch_openfda(client, query, max_results, reaction_filter=None):
    async def _search(drug_q, reaction_q):
        if reaction_q:
            search = (
                f'patient.drug.medicinalproduct:"{drug_q}"'
                f' AND patient.reaction.reactionmeddrapt:"{reaction_q}"'
            )
        else:
            search = f'patient.drug.medicinalproduct:"{drug_q}"'
        params = {"search": search, "limit": max_results}
        return await _openfda_get_with_retry(client, params)

    drug_term = query
    resp = await _search(query, reaction_filter)

    if resp.status_code == 404 and not reaction_filter and len(query.split()) >= 2:
        words = query.split()
        drug_term, reaction_term = words[0], " ".join(words[1:])
        print(f"openFDA: no match for drug phrase '{query}'; falling back to "
              f"drug='{drug_term}' AND reaction='{reaction_term}'")
        resp = await _search(drug_term, reaction_term)

    if resp.status_code == 404:
        # Last-resort: drop the reaction clause and search the drug alone so
        # we still return adverse-event data instead of failing the whole source
        print(f"openFDA: combined query still 404; falling back to drug-only "
              f"search for '{drug_term}'")
        resp = await _search(drug_term, None)

    resp.raise_for_status()
    data = resp.json()

    records = []
    for report in data.get("results", []):
        try:
            safety_id = report.get("safetyreportid", "")
            if not safety_id:
                continue

            receive_date = report.get("receiptdate", "")
            serious = _parse_bool(report.get("serious"))

            reactions = []
            for r in report.get("patient", {}).get("reaction", []):
                reaction = r.get("reactionmeddrapt", "").strip()
                if reaction:
                    reactions.append(reaction)

            query_lower = drug_term.lower()
            drug_entries = []
            searched_drug_roles = []
            searched_drug_names = []
            for drug in report.get("patient", {}).get("drug", []):
                drug_name = drug.get("medicinalproduct", "").strip()
                role = drug.get("drugcharacterization")
                if drug_name:
                    drug_entries.append((drug_name, role))
                    if query_lower in drug_name.lower():
                        searched_drug_roles.append(role)
                        searched_drug_names.append(drug_name.upper())

            searched_drug_display = searched_drug_names[0] if searched_drug_names else drug_term.upper()
            primary_suspects = [f"{n} (role={r})" for n, r in drug_entries if r == "1"]
            other_drugs = [f"{n} (role={r})" for n, r in drug_entries if r != "1"]

            is_primary_suspect = "1" in searched_drug_roles

            if not reactions and not drug_entries:
                continue

            if is_primary_suspect:
                text = (
                    f"PRIMARY SUSPECT DRUG: {searched_drug_display}. "
                    f"Reactions reported: {', '.join(reactions)}. "
                    f"Other co-involved drugs: {', '.join(other_drugs)}."
                )
            else:
                text = (
                    f"CONCOMITANT DRUG: {searched_drug_display} (not primary suspect). "
                    f"Reactions reported: {', '.join(reactions)}. "
                    f"Primary suspect drug(s): {', '.join(primary_suspects)}. "
                    f"Other co-involved drugs: {', '.join(other_drugs)}."
                )

            suspect_label = "primary suspect" if is_primary_suspect else "concomitant"
            metadata = {
                "receive_date": receive_date,
                "serious": serious,
                "reactions": reactions,
                "drugs": [f"{n} (role={r})" for n, r in drug_entries],
                "is_primary_suspect": is_primary_suspect,
            }

            records.append({
                "id": safety_id,
                "source": "openfda",
                "type": "adverse_event",
                "title": f"openFDA adverse event report {safety_id} - {searched_drug_display} ({suspect_label})",
                "text": text,
                "publish_date": None,
                "last_updated": receive_date,
                "url": "https://open.fda.gov/drug/event/",
                "metadata_json": json.dumps(metadata),
            })
        except Exception as e:
            print(f"WARNING: error parsing openfda report: {e}")
    return records


# -----------------------------
# Ingestion orchestrator
# -----------------------------
def _embed_text(embedding_model, text):
    return next(embedding_model.embed([text])).tolist()


async def ingest_topic(topic, max_results=10):
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")

    async with httpx.AsyncClient(timeout=60.0) as client:
        results = await asyncio.gather(
            fetch_pubmed(client, topic, max_results),
            fetch_clinicaltrials(client, topic, max_results),
            fetch_openfda(client, topic, max_results),
            return_exceptions=True,
        )
    for idx, (source_name, result) in enumerate(
        zip(("pubmed", "clinicaltrials", "openfda"), results)
    ):
        if isinstance(result, Exception):
            print(f"WARNING: {source_name} fetch failed and was skipped: {result}")
            results[idx] = []
    fetched = [record for sublist in results for record in sublist]

    counts = {
        "pubmed": len([r for r in fetched if r["source"] == "pubmed"]),
        "clinicaltrials": len([r for r in fetched if r["source"] == "clinicaltrials"]),
        "openfda": len([r for r in fetched if r["source"] == "openfda"]),
    }
    for source, count in counts.items():
        print(f"{source}: fetched {count} records")

    init_db()
    session = get_session()
    try:
        qdrant = AsyncQdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        collections = await qdrant.get_collections()
        names = [c.name for c in collections.collections]
        if QDRANT_COLLECTION not in names:
            await qdrant.create_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
            )
            print(f"Created Qdrant collection '{QDRANT_COLLECTION}'")
        else:
            print(f"Qdrant collection '{QDRANT_COLLECTION}' already exists")

        print(f"Loading embedding model '{EMBED_MODEL}'...")

        await qdrant.create_payload_index(
            collection_name=QDRANT_COLLECTION,
            field_name="source",
            field_schema=PayloadSchemaType.KEYWORD,
        )
        print("Created keyword payload index on 'source' field")
        embedding_model = TextEmbedding(model_name=EMBED_MODEL)
        print("Embedding model ready.")

        inserted = 0
        upserted = 0
        skipped = 0

        for record in fetched:
            try:
                if not record.get("text") or not str(record["text"]).strip():
                    print(f"SKIP (empty text): {record['id']}")
                    skipped += 1
                    continue

                existing = session.query(MedicalRecord).filter_by(
                    id=record["id"], source=record["source"]
                ).first()
                if existing:
                    print(f"SKIP (duplicate): {record['id']} ({record['source']})")
                    skipped += 1
                    continue

                session.add(MedicalRecord(
                    id=record["id"],
                    source=record["source"],
                    type=record["type"],
                    title=record["title"],
                    text=record["text"],
                    publish_date=record["publish_date"],
                    last_updated=record["last_updated"],
                    url=record["url"],
                    metadata_json=record["metadata_json"],
                    ingestion_date=now_str(),
                ))
                session.commit()
                inserted += 1

                vector = await asyncio.to_thread(_embed_text, embedding_model, record["text"])

                payload = dict(record)
                payload["topic"] = topic
                point = PointStruct(
                    id=_point_id(record["id"], record["source"]),
                    vector=vector,
                    payload=payload,
                )
                await qdrant.upsert(collection_name=QDRANT_COLLECTION, points=[point])
                upserted += 1
            except Exception as e:
                print(f"WARNING: failed to ingest record {record['id']} ({record['source']}): {e}")
                skipped += 1

        await qdrant.close()

        print(f"\nSUMMARY: fetched={len(fetched)} inserted_sqlite={inserted} "
              f"upserted_qdrant={upserted} skipped={skipped}")
    finally:
        session.close()


async def main():
    start = time.time()
    init_db()
    await ingest_topic("metformin", max_results=10)
    print(f"Total time: {time.time() - start:.2f} seconds")


if __name__ == "__main__":
    asyncio.run(main())