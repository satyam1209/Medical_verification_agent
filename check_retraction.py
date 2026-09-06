"""Check whether a PubMed article (by PMID) has been retracted.

Uses NCBI ID converter to resolve the DOI for a PMID, then queries
Crossref to look for retraction indicators (is-retracted-by relation
or update-to entries of type 'retraction').

Run: python check_retraction.py
"""

import asyncio
import os

import httpx

IDCONV_URL = "https://pmc.ncbi.nlm.nih.gov/tools/idconv/api/v1/articles/"
ESUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
CROSSREF_URL = "https://api.crossref.org/works/"


def _ncbi_params(base_params):
    api_key = os.getenv("NCBI_API_KEY")
    if api_key:
        base_params = {**base_params, "api_key": api_key, "tool": "medverify"}
    return base_params


async def _get_with_retry(client, url, params=None, attempts=4):
    """GET with exponential backoff on HTTP 429 (NCBI rate limiting)."""
    for attempt in range(1, attempts + 1):
        resp = await client.get(url, params=params)
        if resp.status_code != 429 or attempt == attempts:
            return resp
        delay = 2 ** attempt
        print(f"check_retraction: got 429 from {url}; retrying in {delay}s")
        await asyncio.sleep(delay)
    return resp


async def check_retraction_status(pmid: str):
    retraction_notice_url = None
    note = None

    async with httpx.AsyncClient(timeout=30.0) as client:
        # 1. Resolve PMID -> DOI via NCBI ID converter
        resp = await _get_with_retry(
            client, IDCONV_URL, _ncbi_params({"ids": pmid, "format": "json"})
        )
        resp.raise_for_status()
        records = resp.json().get("records", [])

        if not records:
            return {
                "pmid": pmid,
                "is_retracted": False,
                "retraction_notice_url": retraction_notice_url,
                "note": "unable to verify: no record found for PMID in NCBI ID converter",
            }

        doi = records[0].get("doi")
        if not doi:
            # idconv only covers PMC-indexed articles; fall back to
            # NCBI esummary, which includes a DOI in its articleids list
            resp = await _get_with_retry(
                client,
                ESUMMARY_URL,
                _ncbi_params({"db": "pubmed", "id": pmid, "retmode": "json"}),
            )
            resp.raise_for_status()
            result = resp.json().get("result", {}).get(pmid, {})
            doi = next(
                (a.get("value") for a in result.get("articleids", [])
                 if a.get("idtype") == "doi"),
                None,
            )
        if not doi:
            return {
                "pmid": pmid,
                "is_retracted": False,
                "retraction_notice_url": retraction_notice_url,
                "note": "unable to verify: no DOI found for PMID",
            }

        # 2. Look up the work metadata on Crossref
        resp = await _get_with_retry(client, f"{CROSSREF_URL}{doi}")
        if resp.status_code != 200:
            return {
                "pmid": pmid,
                "is_retracted": False,
                "retraction_notice_url": retraction_notice_url,
                "note": f"unable to verify: Crossref returned HTTP {resp.status_code}",
            }
        data = resp.json().get("message", {})

    # 3. Collect candidate retraction-notice DOIs from the relation field
    #    (is-retracted-by) and the update-to field (type == "retraction")
    candidate_notice_dois = []
    relation = data.get("relation", {})
    for rel in relation.get("is-retracted-by", []):
        if rel.get("id-type") == "doi" and rel.get("id"):
            candidate_notice_dois.append(rel["id"])
    for update in data.get("update-to", []):
        if update.get("type") == "retraction" and update.get("DOI"):
            candidate_notice_dois.append(update["DOI"])

    if not candidate_notice_dois:
        # 3c. Edge case: the work itself is a retraction notice
        if data.get("type") == "retraction":
            return {
                "pmid": pmid,
                "is_retracted": False,
                "retraction_notice_url": retraction_notice_url,
                "note": "this record itself is a retraction notice (type='retraction')",
            }
        return {
            "pmid": pmid,
            "is_retracted": False,
            "retraction_notice_url": retraction_notice_url,
            "note": "no retraction indicators found in Crossref metadata",
        }

    # Crossref's update-to tagging can be noisy (commentaries, expressions of
    # concern tagged as 'retraction'), so prefer a candidate whose own
    # Crossref title reads as a formal retraction NOTICE (starts with
    # "Retraction"/"Retraction-"/"Retraction and"), rather than an article
    # whose own metadata is merely labeled "RETRACTED:"
    notice_doi = candidate_notice_dois[0]
    async with httpx.AsyncClient(timeout=30.0) as client:
        for candidate in candidate_notice_dois:
            resp = await _get_with_retry(client, f"{CROSSREF_URL}{candidate}")
            if resp.status_code != 200:
                continue
            title = (resp.json().get("message", {}) or {}).get("title", [""])[0]
            lower = title.lower()
            if lower.startswith(("retraction", "retraction-", "retraction and")):
                notice_doi = candidate
                break

    return {
        "pmid": pmid,
        "is_retracted": True,
        "retraction_notice_url": f"https://doi.org/{notice_doi}",
        "note": "marked as retracted in Crossref (is-retracted-by / update-to)",
    }


async def main():
    test_pmids = [
        "32450107",   # Surgisphere / hydroxychloroquine Lancet paper (retracted)
        "42696402",   # metformin lactic acidosis case report (normal)
    ]

    for pmid in test_pmids:
        result = await check_retraction_status(pmid)
        print(f"\nPMID {pmid}")
        for key, value in result.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    asyncio.run(main())