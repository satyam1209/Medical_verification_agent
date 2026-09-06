"""Evidence-based-medicine helpers: study-design classification, effect-size/CI
extraction, negative-result detection, PICO parsing, overlap grouping, and an
evidence profile summarizer used to ground the LLM's confidence grading.

Design labels are heuristic (title/abstract classifiers) because the PubMed
summary feed does not expose publication types.
"""

import difflib
import re

DESIGN_RANK = {
    "SYSTEMATIC REVIEW / META-ANALYSIS": 5,
    "RCT (published)": 4,
    "PROSPECTIVE COHORT": 3,
    "RETROSPECTIVE COHORT": 2,
    "CASE-CONTROL": 2,
    "CROSS-SECTIONAL": 2,
    "CLINICAL TRIAL (registry)": 2,
    "CASE REPORT / CASE SERIES": 1,
    "NARRATIVE REVIEW": 1,
    "ADVERSE EVENT REPORT": 1,
    "UNKNOWN": 0,
}

_META_RE = re.compile(
    r"meta[- ]analysis|systematic review|pooled analysis|cochrane|network meta[- ]analysis",
    re.IGNORECASE,
)
_RCT_RE = re.compile(
    r"randomi[sz]ed\s+(?:controlled\s+)?trial|randomi[sz]ed\s+controlled|double[- ]?blind",
    re.IGNORECASE,
)
_PROSP_RE = re.compile(r"prospective", re.IGNORECASE)
_RETRO_RE = re.compile(r"retrospective", re.IGNORECASE)
_CASECTRL_RE = re.compile(r"case[- ]control", re.IGNORECASE)
_CROSS_RE = re.compile(r"cross[- ]sectional", re.IGNORECASE)
_CASE_RE = re.compile(
    r"case report|case series|a case of|case-study", re.IGNORECASE
)
_REVIEW_RE = re.compile(r"\breview\b", re.IGNORECASE)

_SAFETY_RE = re.compile(
    r"adverse|side effect|safety|toxicity|toxic|nephrotox|cardiotox|"
    r"lactic acidosis|serious events|mortality risk",
    re.IGNORECASE,
)
_NEGATIVE_RE = re.compile(
    r"no significant difference|no statistically significant|did not reduce|"
    r"no benefit|no effect on|not found to|failed to show|was not associated|"
    r"no association|absence of benefit|little to no benefit|did not improve|"
    r"no improvement",
    re.IGNORECASE,
)

_STAT_RE = re.compile(
    r"(?P<kind>RR|OR|HR|aOR|aHR|SMD|RD|ARR)\s*[=:]?\s*"
    r"(?P<est>\d+\.\d+)\s*(?:\(\s*(?:95%?\s*CI|CI)\s*"
    r"(?P<low>\d+\.\d+)\s*(?:-|–|to)\s*(?P<high>\d+\.\d+)\s*\))*",
    re.IGNORECASE,
)
_N_RE = re.compile(r"\bn\s*[=:]\s*([\d,]{2,})", re.IGNORECASE)
_P_RE = re.compile(r"\bp\s*(?:=|<|>)?\s*([\d.]+)", re.IGNORECASE)


def design_of(source, title, text, rtype=None):
    if source == "openfda":
        return "ADVERSE EVENT REPORT"
    if rtype == "trial" or source == "clinicaltrials":
        return "CLINICAL TRIAL (registry)"

    hay = f"{title} {text}"
    if _META_RE.search(hay):
        return "SYSTEMATIC REVIEW / META-ANALYSIS"
    if _RCT_RE.search(hay):
        return "RCT (published)"
    if _PROSP_RE.search(hay):
        return "PROSPECTIVE COHORT"
    if _RETRO_RE.search(hay):
        return "RETROSPECTIVE COHORT"
    if _CASECTRL_RE.search(hay):
        return "CASE-CONTROL"
    if _CROSS_RE.search(hay):
        return "CROSS-SECTIONAL"
    if _CASE_RE.search(hay):
        return "CASE REPORT / CASE SERIES"
    if _REVIEW_RE.search(hay):
        return "NARRATIVE REVIEW"
    return "UNKNOWN"


def effect_stats(text):
    stats = {"effect_size": None, "ci": None, "sample_size": None, "p_value": None}
    m = _STAT_RE.search(text or "")
    if m:
        stats["effect_size"] = m.groupdict()
        if m.group("low") and m.group("high"):
            stats["ci"] = (m.group("low"), m.group("high"))
    n = _N_RE.search(text or "")
    if n:
        stats["sample_size"] = n.group(1).rstrip(",.-")
    p = _P_RE.search(text or "")
    if p:
        stats["p_value"] = p.group(1)
    return stats


def is_negative(text):
    return bool(_NEGATIVE_RE.search(text or ""))


def evidence_kind(source, title, text):
    if source == "openfda":
        return "safety (adverse event report)"
    if source == "clinicaltrials":
        return "trial registry"
    if _SAFETY_RE.search(f"{title} {text}"):
        return "safety"
    return "efficacy (or outcomes)"


def characterize_record(record):
    source = record.get("source", "")
    title = record.get("title") or ""
    text = record.get("text") or ""
    rtype = record.get("type")

    design = design_of(source, title, text, rtype)
    stats = effect_stats(text)
    neg = is_negative(f"{title} {text}")
    return {
        "design": design,
        "design_rank": DESIGN_RANK.get(design, 0),
        "stats": stats,
        "negative_result": neg,
        "evidence_kind": evidence_kind(source, title, text),
    }


def _norm_title(t):
    return re.sub(r"[^a-z0-9]", "", (t or "").lower())


def group_overlaps(results):
    """Group indices of near-duplicate / overlapping analyses (same source,
    title similarity >= 0.82). Used to avoid treating overlapping reviews as
    independent evidence."""
    titles = [_norm_title(r.get("title")) for r in results]
    n = len(titles)
    groups = []
    used = set()
    for i in range(n):
        if i in used or not titles[i]:
            continue
        cluster = [i]
        for j in range(i + 1, n):
            if j in used or not titles[j]:
                continue
            if titles[i] == titles[j]:
                cluster.append(j)
                continue
            ratio = difflib.SequenceMatcher(None, titles[i], titles[j]).ratio()
            if ratio >= 0.82:
                cluster.append(j)
        if len(cluster) > 1:
            groups.append(cluster)
            used.update(cluster)
        elif len(cluster) == 1 and titles[i]:
            used.add(i)
    return groups


_POP_RE = re.compile(
    r"\bin\s+((?:older|elderly|adult|children|children? and adolescents|"
    r"infants|pregnant|postmenopausal|diabetic|patients with|people with|women|men)"
    r"[a-z0-9 ,\-]{0,40})",
    re.IGNORECASE,
)
_OUTCOME_VERB_RE = re.compile(
    r"(?:reduce|prevent|treat|improve|lower|decrease|increase|affect|mitigate|cause|"
    r"induce|lead to)\s+"
    r"((?:[a-z0-9\- ]{2,45}?))(?:\s+in\s|$|\?)",
    re.IGNORECASE,
)
_CMP_RE = re.compile(
    r"(?:compared?\s+(?:with|to)|versus|vs\.?)\s+([a-z0-9 ,\-]{2,30})",
    re.IGNORECASE,
)
_INTERVENTION_RES = re.compile(
    r"(?:does|can|is|effect of|role of|use of|benefit of)\s+"
    r"((?:[a-z0-9\- ]{2,45}?))(?:\s+(?:reduce|prevent|treat|improve|lower|affect))",
    re.IGNORECASE,
)


def parse_pico(query):
    q = query or ""
    pico = {"population": None, "intervention": None, "comparator": None, "outcome": None}
    pm = _POP_RE.search(q)
    if pm:
        pico["population"] = pm.group(1).strip()
    im = _INTERVENTION_RES.search(q)
    if im:
        pico["intervention"] = im.group(1).strip()
    else:
        mm = re.search(
            r"\b(metformin|statins?|aspirin|zinc|probiotics?|vitamin d|calcium|"
            r"omega-3|omega-3 fatty acids?)\b", q, re.IGNORECASE)
        if mm:
            pico["intervention"] = mm.group(0).strip()
    cm = _CMP_RE.search(q)
    if cm:
        pico["comparator"] = cm.group(1).strip()
    om = _OUTCOME_VERB_RE.search(q)
    if om:
        pico["outcome"] = om.group(1).strip()
    return pico


def evidence_profile(results):
    """Build a compact EBM profile of the retrieved set for the LLM prompt."""
    lines = []
    designs = [r.get("design") for r in results]
    profile = {
        "n": len(results),
        "designs": designs,
        "any_negative": any(r.get("negative_result") for r in results),
        "negatives": [
            f"Source {i}: {r.get('title')}"
            for i, r in enumerate(results, 1)
            if r.get("negative_result")
        ],
        "with_ci": sum(1 for r in results if r.get("stats", {}).get("ci")),
        "safety": sum(1 for r in results if "safety" in (r.get("evidence_kind") or "")),
    }

    lines.append(f"Retrieved evidence lines: {profile['n']} "
                 f"(designs: {', '.join(designs) or 'none'})")
    lines.append(
        f"Precision: {profile['with_ci']} source(s) report a 95% CI for the effect; "
        f"heterogeneity/subgroups only if stated in the excerpt."
    )
    lines.append(f"Negative/null findings reported in: "
                 + (", ".join(profile["negatives"]) if profile["negatives"] else "none"))
    lines.append(
        f"Safety vs efficacy: {profile['safety']} source(s) are safety/adverse-event "
        f"related."
    )
    return "\n".join(lines)


if __name__ == "__main__":
    import json
    from retrieve import search_kb

    results = search_kb("does metformin cause vitamin B12 deficiency?", top_k=8, truncate=False)
    for i, r in enumerate(results, 1):
        p = characterize_record(r)
        print(f"{i}. [{r['source']}] design={p['design']} neg={p['negative_result']} "
              f"kind={p['evidence_kind']} stats={p['stats']} | {r['title'][:60]}")
    print("\nOverlap groups:", group_overlaps(results))
    print("\nProfile:\n" + evidence_profile(results))
    print("\nPICO:", parse_pico("does metformin cause vitamin B12 deficiency in adults?"))