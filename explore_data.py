"""
Explore detailed data from all 3 data sources:
1. PubMed (via Entrez/biopython) - article details
2. ClinicalTrials.gov API - study details
3. openFDA API - adverse event report details

Run: python explore_data.py
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

# -----------------------------
# 1. Explore PubMed article
# -----------------------------
def explore_pubmed():
    print("\n=== 1. PubMed Article ===")
    from Bio import Entrez

    Entrez.email = os.getenv("NCBI_EMAIL", "test@example.com")
    api_key = os.getenv("NCBI_API_KEY")
    if api_key:
        Entrez.api_key = api_key

    handle = Entrez.esearch(db="pubmed", term="metformin long term side effects", retmax=1)
    record = Entrez.read(handle)
    handle.close()

    ids = record["IdList"]
    if not ids:
        print("No articles found")
        return

    pmid = ids[0]
    handle = Entrez.efetch(db="pubmed", id=pmid, rettype="medline", retmode="xml")
    article = Entrez.read(handle)
    handle.close()

    art = article["PubmedArticle"][0]
    medline = art.get("MedlineCitation", {})
    article_meta = medline.get("Article", {})

    title = article_meta.get("ArticleTitle", "N/A")
    journal = article_meta.get("Journal", {}).get("Title", "N/A")

    pub_date = "N/A"
    journal_issue = article_meta.get("Journal", {}).get("JournalIssue", {})
    pub_date_info = journal_issue.get("PubDate", {})
    for key in ("Year", "Month", "Day"):
        if pub_date_info.get(key):
            pub_date = pub_date_info.get(key)
            break

    authors = article_meta.get("AuthorList", [])
    author_names = []
    for a in authors:
        last = a.get("LastName", "")
        fore = a.get("ForeName", "")
        initial = a.get("Initials", "")
        name = " ".join(part for part in (fore, initial, last) if part)
        author_names.append(name if name else "N/A")

    abstract_text = "N/A"
    abstract = article_meta.get("Abstract", {})
    abstract_text_parts = abstract.get("AbstractText", [])
    if abstract_text_parts:
        parts = abstract_text_parts if isinstance(abstract_text_parts, list) else [abstract_text_parts]
        abstract_text = " ".join(str(p) for p in parts if str(p))

    print(f"PMID: {pmid}")
    print(f"Title: {title}")
    print(f"Journal: {journal}")
    print(f"Publication date: {pub_date}")
    print(f"Authors: {', '.join(author_names)}")
    print(f"Abstract:\n{abstract_text}")


# -----------------------------
# 2. Explore ClinicalTrials.gov study
# -----------------------------
def explore_clinicaltrials():
    print("\n=== 2. ClinicalTrials.gov Study ===")
    url = "https://clinicaltrials.gov/api/v2/studies"
    params = {
        "query.term": "metformin",
        "pageSize": 1
    }
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    data = resp.json()
    studies = data.get("studies", [])
    if not studies:
        print("No studies found")
        return

    study = studies[0]
    proto = study.get("protocolSection", {})

    ident = proto.get("identificationModule", {})
    nct_id = ident.get("nctId", "N/A")
    title = ident.get("briefTitle", "N/A")

    status_module = proto.get("statusModule", {})
    status = status_module.get("overallStatus", "N/A")
    start_date = status_module.get("startDateStruct", {}).get("date", "N/A")
    last_update = status_module.get("lastUpdatePostDateStruct", {}).get("date", "N/A")

    conditions = ", ".join(proto.get("conditionsModule", {}).get("conditions", [])) or "N/A"

    design = proto.get("designModule", {})
    phases = ", ".join(design.get("phases", [])) or "N/A"

    desc = proto.get("descriptionModule", {})
    summary = desc.get("briefSummary", "N/A")

    print(f"NCT ID: {nct_id}")
    print(f"Title: {title}")
    print(f"Status: {status}")
    print(f"Start date: {start_date}")
    print(f"Last update date: {last_update}")
    print(f"Conditions: {conditions}")
    print(f"Phase: {phases}")
    print(f"Brief summary:\n{summary}")


# -----------------------------
# 3. Explore openFDA adverse event report
# -----------------------------
def explore_openfda():
    print("\n=== 3. openFDA Adverse Event Report ===")
    url = "https://api.fda.gov/drug/event.json"
    params = {
        "search": 'patient.drug.medicinalproduct:"metformin"',
        "limit": 1
    }
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    data = resp.json()
    results = data.get("results", [])
    if not results:
        print("No adverse event reports found")
        return

    report = results[0]
    safety_id = report.get("safetyreportid", "N/A")
    receive_date = report.get("receivedate", "N/A")

    seriousness = []
    for key in ("serious", "seriousnesscongenitalanomali", "seriousnessdeath",
                "seriousnessdisabling", "seriousnesshospitalization",
                "seriousnesslifethreatening", "seriousnessother"):
        if report.get(key) == "1":
            seriousness.append(key.replace("seriousness", "serious").replace("congenitalanomali", "congenital anomaly")
                               .replace("death", "death").replace("disabling", "disabling")
                               .replace("hospitalization", "hospitalization")
                               .replace("lifethreatening", "life threatening").replace("other", "other"))
    serious_flag = "Yes" if report.get("serious") == "1" else "No"
    seriousness_str = ", ".join(seriousness) if seriousness else "Not reported"

    drugs = []
    for drug in report.get("patient", {}).get("drug", []):
        name = drug.get("medicinalproduct", "N/A")
        active = drug.get("openfda", {}).get("brand_name", [])
        role = drug.get("drugcharacterization", "N/A")
        drugs.append(f"{name} (role={role}, brand={', '.join(active) if active else 'N/A'})")

    reactions = []
    for r in report.get("patient", {}).get("reaction", []):
        reactions.append(r.get("reactionmeddrapt", "N/A"))

    print(f"Safety report ID: {safety_id}")
    print(f"Receive date: {receive_date}")
    print(f"Serious: {serious_flag} ({seriousness_str})")
    print(f"Drugs involved:")
    for d in drugs:
        print(f"  - {d}")
    print(f"Reactions: {reactions}")


if __name__ == "__main__":
    try:
        explore_pubmed()
    except Exception as e:
        print(f"PubMed exploration failed: {e}")

    try:
        explore_clinicaltrials()
    except Exception as e:
        print(f"ClinicalTrials.gov exploration failed: {e}")

    try:
        explore_openfda()
    except Exception as e:
        print(f"openFDA exploration failed: {e}")