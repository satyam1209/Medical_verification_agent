"""Streamlit UI for the medical evidence verification & grading pipeline.

Runs the full pipeline with visible progress:
  1. Searches supported datasets (PubMed, ClinicalTrials.gov, openFDA)
     and reports how many records each one returned
  2. Retrieves the most relevant evidence from the knowledge base
  3. Re-checks retraction / freshness of the retrieved pubs
  4. Assembles the numbered evidence context
  5. Streams the final LLM answer

Run: streamlit run app.py
"""

import asyncio
import contextlib
import io

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from ingest import DEFAULT_TTL_HOURS, ingest_topic
from retrieve import clear_search_cache, search_kb
from synthesize import build_user_message, synthesize_answer_stream
from evidence import outcome_category, pico_component_coverage, parse_pico

st.set_page_config(page_title="MedVerify", page_icon=":medical_symbol:", layout="centered")

st.title("MedVerify — Medical Evidence Verifier")
st.caption(
    "Question the evidence, not the echo. Sources: PubMed, ClinicalTrials.gov, "
    "openFDA adverse event reports. Answers are grounded strictly in the "
    "ingested evidence and graded by confidence."
)

query = st.text_input(
    "Ask a medical question",
    placeholder="e.g. does metformin cause vitamin B12 deficiency?",
)
run = st.button("Run verification", type="primary", use_container_width=True)

with st.expander("Data freshness"):
    st.caption(
        "Knowledge bases refresh automatically: a topic re-pulls recent studies "
        f"once it is older than the {DEFAULT_TTL_HOURS}h TTL, so answers never rely "
        "on a stale snapshot. Tick the box below to force a full refresh now."
    )
    force_refresh = st.checkbox("Force full refresh from sources for this query",
                                value=False)


def run_pipeline(q, refresh=False):
    with st.status("Running the evidence pipeline…", expanded=True) as status:

        # ---- STEP 1: ingestion ------------------------------------------------
        st.write("**1. Updating knowledge base**")
        summary = asyncio.run(
            ingest_topic(q, max_results=15, force_refresh=refresh)
        )
        if summary.get("cached"):
            st.write(
                f"  - Topic knowledge base already current "
                f"(ingested {summary['last_ingested_at']}); cached evidence used "
                f"— data is younger than the {DEFAULT_TTL_HOURS}h TTL"
            )
        else:
            since_note = (
                f" (incremental since {summary['since']})"
                if summary.get("mode") == "incremental"
                else " (full)"
            )
            st.write(f"  - Refreshing sources{since_note}")
            for source_name, count in summary["fetched"].items():
                st.write(f"  - {source_name}: fetched **{count}** records")
        st.write(
            f"  - Knowledge base state: {summary['inserted']} new, "
            f"{summary['skipped']} already present"
        )
        if refresh:
            clear_search_cache()

        # ---- STEP 2: retrieval ------------------------------------------------
        st.write("**2. Retrieving most relevant evidence**")
        stdout_capture = io.StringIO()
        with contextlib.redirect_stdout(stdout_capture):
            results = search_kb(q, top_k=8, truncate=False)
        st.write(f"  - Retrieved **{len(results)}** evidence items from the KB")

        retraction_warnings = [
            line for line in stdout_capture.getvalue().splitlines()
            if line.startswith("WARNING: Excluded retracted paper")
        ]
        for warning in retraction_warnings:
            st.write(f"  - :red-badge[**{warning}**]")

        # ---- STEP 3: freshness / design review --------------------------------
        st.write("**3. Reviewing freshness & study design of retrieved evidence**")
        for r in results:
            pub = r.get("publish_date") or "unknown date"
            st.write(
                f"  - [{r['source']}] {r['design']} | {r['freshness_label']} "
                f"({pub}) | {r['title'][:70]}"
            )

        # ---- STEP 4: context assembly -----------------------------------------
        st.write("**4. Assembling evidence context (numbered sources)**")
        user_msg = build_user_message(q, top_k=8)
        num_sources = user_msg.split("---\n")[0].count("[Source ")
        st.write(f"  - Context ready with **{num_sources}** numbered sources")

        pico = parse_pico(q)
        coverage = pico_component_coverage(pico, results)
        pico_line = "PICO: " + ", ".join(
            f"{k}={v or 'n/a'}" for k, v in pico.items()
        )
        cover_line = "Coverage (keyword hint): " + ", ".join(
            f"{k}={'touched' if v else 'NO source' if v is False else 'n/a'}"
            for k, v in coverage.items()
        )
        st.write(
            f"  - {pico_line}  |  {cover_line}\n"
            f"  - Outcome type: {outcome_category(pico.get('outcome'))}"
        )

        status.update(label="Synthesis complete", state="complete", expanded=False)

    # ---- STEP 5: streaming model response --------------------------------------
    st.write("**5. Model response**")
    st.write_stream(synthesize_answer_stream(q, top_k=8))


if run and query.strip():
    try:
        run_pipeline(query.strip(), refresh=force_refresh)
    except Exception as exc:
        st.error(f"Pipeline failed: {exc}")
elif run:
    st.warning("Please enter a question first.")