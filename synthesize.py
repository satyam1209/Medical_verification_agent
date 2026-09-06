"""Synthesize answers from medical evidence using Groq LLM + Qdrant knowledge base.

Run: python synthesize.py
"""

import os
import re

from dotenv import load_dotenv

load_dotenv()

from groq import Groq

from retrieve import search_kb
from evidence import group_overlaps, evidence_profile, parse_pico

SYSTEM_PROMPT = """\
You are a medical evidence verifier applying Evidence-Based Medicine (EBM) methodology. You will be given a user question, a PICO decomposition, an evidence profile of the retrieved sources, and numbered source excerpts from PubMed papers, ClinicalTrials.gov trials, and openFDA adverse event reports. Answer the question using ONLY the provided sources.

EBM Principles you MUST follow:
1. Identify PICO (Population, Intervention, Comparator, Outcome) for the question before answering.
2. Weigh evidence by study design hierarchy, not by raw source count: RCTs & systematic reviews/meta-analyses > prospective cohort > retrospective cohort/case-control > case reports/narrative reviews/adverse-event reports.
3. Do NOT treat overlapping or redundant analyses as independent evidence; if two sources are the same or overlapping meta-analyses/trials, count them as ONE line of evidence and say so.
4. Report effect size with 95% CI, sample size, and (when given) absolute measures such as NNT/ARR. Never invent numbers not in the excerpt.
5. Distinguish relative from absolute effects, composite from individual endpoints, and surrogate from clinical outcomes.
6. Explicitly separate EFFICACY evidence from SAFETY evidence; an adverse-event report supports a risk claim, not a benefit claim.
7. In observational studies (cohort, case-control, cross-sectional, AE reports) state association-to-causation limitations; do not claim causation.
8. Report heterogeneity (I-squared), risk of bias, and indirectness IF the excerpt states them; otherwise state they are unavailable.
9. Do not generalize beyond the study population, dose, formulation, drug, or endpoint presented in the excerpts.
10. Explicitly state contradictory or negative evidence when present; do not cherry-pick supporting sources.
11. Base your final confidence label on evidence quality, consistency, precision (CIs), directness, and limitations — NOT on source count alone.
12. Make the final conclusion NO STRONGER or BROADER than the underlying evidence.

Output format:
- Start with a 1-line PICO restatement of the question.
- Body: cite [Source N] for every factual claim.
- Before the confidence label, list: (a) number of INDEPENDENT evidence lines with their designs; (b) consistency/direction across sources; (c) precision (which sources report 95% CIs) and directness; (d) limitations (heterogeneity, bias, indirectness) if reported; (e) efficacy vs safety split.
- End with exactly one of: "Confidence: STRONG EVIDENCE", "Confidence: MODERATE EVIDENCE", or "Confidence: WEAK EVIDENCE", followed by a short justification in parentheses based on (a)-(e)."""

GROQ_MODEL = "openai/gpt-oss-120b"

client = Groq()


def _format_entries(results):
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

        design = r.get("design") or "UNKNOWN"
        flags = []
        if r.get("negative_result"):
            flags.append("NEGATIVE/NULL RESULT")
        if r.get("evidence_kind"):
            flags.append(r["evidence_kind"].upper())
        stats = r.get("stats") or {}
        stat_frag = ""
        if stats.get("effect_size") and stats.get("ci"):
            es = stats["effect_size"]
            lo, hi = stats["ci"]
            stat_frag = f" | {es['kind']} {es['est']} (95% CI {lo}-{hi})"
        if stats.get("sample_size") and stat_frag:
            stat_frag += f" | n={stats['sample_size']}"

        entry = (
            f"[Source {i}] {source_label} | {design} | {freshness_frag}"
            f"{stat_frag} | {r['title']} | {'; '.join(flags) + ' | ' if flags else ''}"
            f"{r['text']} | URL: {r['url']}{suspect_tag}"
        )
        entries.append(entry)
    return entries


def build_context(query: str, top_k: int = 8):
    results = search_kb(query, top_k=top_k, truncate=False)
    entries = _format_entries(results)

    context = "\n\n".join(entries)

    overlap_groups = group_overlaps(results)
    if overlap_groups:
        notices = []
        for group in overlap_groups:
            refs = ", ".join(f"Source {i}" for i in group)
            notices.append(
                f"NOTE: {refs} are overlapping/redundant analyses — treat them as "
                f"ONE independent line of evidence, do not double-count."
            )
        context += "\n\n" + "\n".join(notices)

    return context


def build_user_message(query: str, top_k: int = 8):
    results = search_kb(query, top_k=top_k, truncate=False)
    entries = _format_entries(results)
    context = "\n\n".join(entries)

    override_lines = []
    overlap_groups = group_overlaps(results)
    if overlap_groups:
        for group in overlap_groups:
            refs = ", ".join(f"Source {i}" for i in group)
            override_lines.append(
                f"NOTE: {refs} are overlapping/redundant analyses — treat them as "
                f"ONE independent line of evidence, do not double-count."
            )
        context += "\n\n" + "\n".join(override_lines)

    profile = evidence_profile(results)
    pico = parse_pico(query)
    pico_str = (
        f"PICO: Population={pico['population'] or 'not stated'}; "
        f"Intervention={pico['intervention'] or 'not stated'}; "
        f"Comparator={pico['comparator'] or 'not stated'}; "
        f"Outcome={pico['outcome'] or 'not stated'}"
    )
    return (
        f"{context}\n\n---\n{pico_str}\n\nEvidence profile:\n{profile}\n\n"
        f"Question: {query}"
    )


def synthesize_answer(query: str):
    user_msg = build_user_message(query)

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


def synthesize_answer_stream(query: str, top_k: int = 8):
    """Yield answer text chunks as Groq streams them back."""
    user_msg = build_user_message(query, top_k=top_k)

    print("Sending context to Groq LLM (streaming)...")
    stream = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.2,
        max_tokens=2048,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


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