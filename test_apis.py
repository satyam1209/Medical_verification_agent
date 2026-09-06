"""
Day 1 - Test script to verify access to all data sources:
1. PubMed (via Entrez/biopython)
2. ClinicalTrials.gov API
3. openFDA API

Run: python test_apis.py
"""

import json
import os
import requests
from dotenv import load_dotenv

load_dotenv()

# -----------------------------
# 1. Test PubMed (Entrez)
# -----------------------------
def test_pubmed():
    print("\n--- Testing PubMed ---")
    from Bio import Entrez

    Entrez.email = os.getenv("NCBI_EMAIL", "test@example.com")
    api_key = os.getenv("NCBI_API_KEY")
    if api_key:
        Entrez.api_key = api_key

    handle = Entrez.esearch(db="pubmed", term="metformin long term side effects", retmax=5)
    record = Entrez.read(handle)
    handle.close()

    with open("pubmed.json", "w") as f:
        json.dump(record, f, indent=2)
    print("Saved full esearch response to pubmed.json")

    ids = record["IdList"]
    print(f"Found {len(ids)} PubMed IDs: {ids}")

    if ids:
        fetch_handle = Entrez.efetch(db="pubmed", id=ids[0], rettype="abstract", retmode="text")
        abstract = fetch_handle.read()
        fetch_handle.close()
        with open("pubmed_abstract.json", "w") as f:
            json.dump({"id": ids[0], "abstract": abstract}, f, indent=2)
        print("Saved abstract to pubmed_abstract.json")
        print(f"Sample abstract (first 300 chars):\n{abstract[:300]}")


# -----------------------------
# 2. Test ClinicalTrials.gov
# -----------------------------
def test_clinicaltrials():
    print("\n--- Testing ClinicalTrials.gov ---")
    url = "https://clinicaltrials.gov/api/v2/studies"
    params = {
        "query.term": "metformin",
        "pageSize": 5
    }
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    data = resp.json()
    with open("clinicaltrials.json", "w") as f:
        json.dump(data, f, indent=2)
    print("Saved response to clinicaltrials.json")
    studies = data.get("studies", [])
    print(f"Found {len(studies)} trials")
    if studies:
        first = studies[0]
        title = first.get("protocolSection", {}).get("identificationModule", {}).get("briefTitle", "N/A")
        print(f"Sample trial title: {title}")


# -----------------------------
# 3. Test openFDA
# -----------------------------
def test_openfda():
    print("\n--- Testing openFDA ---")
    url = "https://api.fda.gov/drug/event.json"
    params = {
        "search": 'patient.drug.medicinalproduct:"metformin"',
        "limit": 5
    }
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    data = resp.json()
    with open("openfda.json", "w") as f:
        json.dump(data, f, indent=2)
    print("Saved response to openfda.json")
    results = data.get("results", [])
    print(f"Found {len(results)} adverse event reports")
    if results:
        reactions = results[0].get("patient", {}).get("reaction", [])
        reaction_names = [r.get("reactionmeddrapt") for r in reactions]
        print(f"Sample reactions: {reaction_names}")


if __name__ == "__main__":
    try:
        test_pubmed()
    except Exception as e:
        print(f"PubMed test failed: {e}")

    try:
        test_clinicaltrials()
    except Exception as e:
        print(f"ClinicalTrials.gov test failed: {e}")

    try:
        test_openfda()
    except Exception as e:
        print(f"openFDA test failed: {e}")