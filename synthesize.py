"""Synthesize answers from medical evidence using Groq LLM + Qdrant knowledge base.

Run: python synthesize.py
"""

import os
import re

from dotenv import load_dotenv

load_dotenv()

from groq import Groq

from retrieve import search_kb

SYSTEM_PROMPT = """\
You are a medical evidence verification assistant. You will be given a user question and a set of numbered source excerpts from PubMed papers, ClinicalTrials.gov trials, and openFDA adverse event reports. Your job is to answer the question using ONLY the provided sources. Rules:
1. Cite sources using [Source N] notation for every factual claim.
2. If openFDA sources show a drug as 'CONCOMITANT DRUG (not primary suspect)', explicitly state that this drug was NOT the primary suspect for that reaction, and name the actual primary suspect drug if given.
3. If sources conflict or evidence is thin (e.g. only 1 source, or old data), explicitly say so.
4. Do NOT use any outside knowledge beyond the provided sources.
5. End your answer with a confidence label: STRONG EVIDENCE (multiple consistent sources), MODERATE EVIDENCE (some support but limited), or WEAK EVIDENCE (thin, conflicting, or only concomitant data).
6. Before giving your final confidence label, explicitly list: (a) how many distinct sources support the main claim, (b) whether any source presents a different magnitude, timeframe, or conclusion, and (c) whether the evidence is from case reports (weaker) versus clinical trials or systematic reviews (stronger)."""

GROQ_MODEL = "openai/gpt-oss-120b"

client = Groq()


def build_context(query: str, top_k: int = 8):
    results = search_kb(query, top_k=top_k, truncate=False)

    entries = []
    for i, r in enumerate(results, 1):
        source_label = r["source"].upper()
        meta = r.get("metadata_json")
        suspect_tag = ""
        if meta and isinstance(meta, str):
            import json
            try:
                parsed = json.loads(meta)
                is_ps = parsed.get("is_primary_suspect")
                if is_ps is True:
                    suspect_tag = " [PRIMARY SUSPECT]"
                elif is_ps is False:
                    suspect_tag = " [CONCOMITANT DRUG - NOT PRIMARY SUSPECT]"
            except Exception:
                pass

        label = r.get("freshness_label") or "UNKNOWN"
        pub = r.get("publish_date") or ""
        match = re.match(r"(\d{4})", pub)
        if label and match:
            freshness_frag = f"({label}, published {match.group(1)})"
        else:
            freshness_frag = f"({label})"

        entry = (
            f"[Source {i}] {source_label} | {freshness_frag} | {r['title']} | "
            f"{r['text']} | URL: {r['url']}{suspect_tag}"
        )
        entries.append(entry)

    return "\n\n".join(entries)


def synthesize_answer(query: str):
    context = build_context(query)
    user_msg = f"{context}\n\n---\nQuestion: {query}"

    print("Sending context to Groq LLM...")
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.2,
        max_tokens=2048,
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    print(f"Medical Evidence Synthesis (Groq model: {GROQ_MODEL})")
    print("Type 'exit' to quit.\n")

    while True:
        query = input("Enter query: ").strip()
        if query.lower() == "exit":
            break
        if not query:
            continue

        try:
            answer = synthesize_answer(query)
            print(f"\n{answer}\n")
        except Exception as e:
            print(f"Error: {e}")