"""Synthesize answers from medical evidence using Groq LLM + Qdrant knowledge base.

Run: python synthesize.py
"""

import os
import re

from dotenv import load_dotenv

load_dotenv()

from groq import Groq

from retrieve import search_kb_staged
from evidence import group_overlaps, evidence_profile, parse_pico

SYSTEM_PROMPT = """\
You are a medical evidence verifier applying Evidence-Based Medicine (EBM) methodology. You will be given a user question, a PICO decomposition, an evidence profile, and numbered source excerpts from PubMed papers, ClinicalTrials.gov trials, and openFDA adverse event reports (some retrieved semantically, some via targeted retrieval by evidence type). Answer the question using ONLY the provided sources.

EBM Principles you MUST follow:
1. Weigh evidence by study design hierarchy: RCTs & systematic reviews/meta-analyses > prospective cohort > retrospective cohort/case-control > case reports/narrative reviews/adverse-event reports. A guideline recommendation is advisory, not independent efficacy evidence; a narrative review is not equivalent to an RCT/meta-analysis.
2. Do NOT treat overlapping/redundant analyses as independent evidence; collapse them into ONE independent evidence base and say so.
3. Report effect size with its 95% CI, the effect measure (OR vs RR vs HR vs RD are NOT interchangeable — compare only when outcomes, populations, and timeframes are comparable), sample size, follow-up, and absolute measures (NNT/ARR) when given. Never invent numbers.
4. Keep outcomes distinct: mortality != survival; stage at diagnosis != mortality; surrogate/biomarker != clinical outcome; composite != individual component; modeled/estimated != observed. Prefer evidence matching the requested outcome directly; flag indirect evidence.
5. Separate EFFICACY from SAFETY, and association from causation (observational/AE-report designs support association only).
6. Report heterogeneity (I-squared), risk of bias, precision, directness, publication bias, and applicability ONLY if the excerpt states them; otherwise say they are unavailable.
7. Do not silently extrapolate across population/age, intervention/dose/formulation, comparator, duration, setting, or outcome. If you extrapolate, LABEL it explicitly and say why it is justified.
8. Handle negative evidence precisely — these are not interchangeable: "evidence of no effect" vs "evidence is inconclusive" vs "no direct evidence identified" vs "important evidence missing from retrieval".
9. Confidence must be PICO-specific; it may differ by population, dose, formulation, outcome, or subgroup. Note such differences when the excerpts support them.

CRITICAL RULE — Retrieval completeness is NOT evidence strength:
- "No high-quality evidence retrieved" != "no high-quality evidence exists"; "RCTs not retrieved" != "no RCTs exist".
- Missing evidence is INCOMPLETE RETRIEVAL, reported as an evidence-coverage problem — NEVER automatically down-graded to weak/low evidence.
- Absence of contradictory evidence is NOT evidence of consistency unless retrieval is judged comprehensive.
- If retrieval is partial/incomplete but you cannot rule out better evidence in the wider literature, do NOT downgrade the overall certainty solely because it was not retrieved.

Output format (MUST follow exactly, in this order):
**Bottom line** — 1-2 sentences, the answer to the question as posed.
**Direct evidence** — evidence matching the PICO directly; cite [Source N], give design, effect measure + CI + n when present, and note safety vs efficacy.
**Supporting / mixed evidence** — adjacent, indirect, or contradictory findings with [Source N].
**Limitations** — quality, directness, completeness, heterogeneity/bias (only if stated), applicability.
**Evidence coverage:** COMPLETE|PARTIAL|INCOMPLETE — use the machine coverage hint; it only describes the RETRIEVED set.
**Certainty**:
- Retrieved evidence certainty: STRONG|MODERATE|WEAK — certainty of conclusions drawn directly from the retrieved sources.
- Overall evidence certainty: STRONG|MODERATE|WEAK — your judgment of the TOTAL relevant evidence base. If the retrieved evidence is incomplete/partial for the PICO so that overall certainty cannot be judged, state EXACTLY: "OVERALL CERTAINTY CANNOT BE DETERMINED FROM THE RETRIEVED EVIDENCE."
**Gaps** — evidence types not retrieved (e.g. missing RCTs, systematic reviews, safety data, subgroups), and which targeted searches would help.
Final line — EXACTLY one of:
"Confidence: STRONG EVIDENCE"
"Confidence: MODERATE EVIDENCE"
"Confidence: WEAK EVIDENCE"
"Confidence: OVERALL CERTAINTY CANNOT BE DETERMINED FROM THE RETRIEVED EVIDENCE"
Use the last form when overall certainty truly cannot be judged; otherwise report the overall certainty value."""

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
    results = search_kb_staged(query, top_k=top_k, truncate=False)
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
    results = search_kb_staged(query, top_k=top_k, truncate=False)
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

    pico = parse_pico(query)
    profile = evidence_profile(results, pico=pico)
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