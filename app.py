"""Streamlit app: upload architectural PDF(s), extract drawings + dimensions.

Run with: streamlit run app.py
"""
from __future__ import annotations

import json
import logging
import time

import streamlit as st

from pipeline.config import ConfigError, load_settings, missing_vars
from pipeline.orchestrator import _suggest_missing_callout_number, run_pipeline
from pipeline.schema import ExtractionResult

logging.basicConfig(level=logging.INFO)

st.set_page_config(page_title="Drawing Dimension Extraction", page_icon="📐", layout="wide")

st.title("📐 Drawing Dimension Extraction")
st.caption(
    "Upload PDF(s) of architectural/structural drawings. The pipeline counts the "
    "individual drawings on each sheet, classifies each one, and extracts every "
    "dimension with a unique ID."
)


def render_result(result: ExtractionResult, key_prefix: str) -> None:
    """Render one file's extraction result: metrics, download, suspect-drawing
    checks, and the per-drawing expanders. `key_prefix` keeps widget keys
    unique when this is called once per tab across multiple uploaded files."""
    col1, col2, col3 = st.columns(3)
    col1.metric("Pages", result.total_pages)
    col2.metric("Drawings found", result.total_drawings)
    col3.metric(
        "Total dimensions",
        sum(len(d.dimensions) for d in result.drawings),
    )

    st.download_button(
        "⬇️ Download full result (JSON)",
        data=result.model_dump_json(indent=2),
        file_name=f"{result.source_file.rsplit('.', 1)[0]}_extraction.json",
        mime="application/json",
        key=f"{key_prefix}-download-full",
    )

    # Surface the same suspect-drawing checks the pipeline logs, right in the
    # UI: a duplicate title on the same page, or a drawing with nothing
    # extracted, are the two patterns that have shown up as real mistakes
    # (a mislabeled drawing, or a title block/legend mistaken for a drawing).
    suspect_notes: list[str] = []
    titles_by_page: dict[int, dict[str, str]] = {}
    callout_numbers_by_page: dict[int, dict[str, str]] = {}
    for d in result.drawings:
        title = d.drawing_metadata.drawing_title
        if title:
            seen = titles_by_page.setdefault(d.page_number, {})
            if title in seen:
                suspect_notes.append(
                    f"**{seen[title]}** and **{d.drawing_id}** (page {d.page_number}) share the "
                    f"same title *\"{title}\"* — check they weren't mislabeled."
                )
            else:
                seen[title] = d.drawing_id
        callout_number = d.drawing_metadata.title_callout_number
        if callout_number:
            seen_callouts = callout_numbers_by_page.setdefault(d.page_number, {})
            if callout_number in seen_callouts:
                page_drawings = [dr for dr in result.drawings if dr.page_number == d.page_number]
                suggestion = _suggest_missing_callout_number(page_drawings)
                suggestion_text = (
                    f" The only number missing from this page's 1..{len(page_drawings)} "
                    f"sequence is *\"{suggestion}\"* — likely the real value for one of them."
                    if suggestion else ""
                )
                suspect_notes.append(
                    f"**{seen_callouts[callout_number]}** and **{d.drawing_id}** (page {d.page_number}) "
                    f"report the same title-callout number *\"{callout_number}\"* — one of them likely "
                    f"read the wrong drawing's title callout.{suggestion_text}"
                )
            else:
                seen_callouts[callout_number] = d.drawing_id
        if not d.dimensions and not d.elevation_datums:
            suspect_notes.append(
                f"**{d.drawing_id}** has no dimensions or elevation datums at all — could be a "
                "title block/legend/keyplan mistakenly segmented as its own drawing."
            )
    if suspect_notes:
        with st.expander(f"⚠️ {len(suspect_notes)} drawing(s) flagged for a manual check", expanded=True):
            for note in suspect_notes:
                st.markdown(f"- {note}")

    st.subheader("Drawings")
    for drawing in result.drawings:
        meta = drawing.drawing_metadata
        header = f"{drawing.drawing_id} — {drawing.drawing_type} (page {drawing.page_number})"
        if meta.drawing_title:
            header += f" — {meta.drawing_title}"
        if drawing.detail_pass_applied:
            header += " 🔍"
        with st.expander(header):
            meta_bits = [
                (label, value)
                for label, value in [
                    ("Title callout #", meta.title_callout_number),
                    ("Project", meta.project_name),
                    ("Sheet no.", meta.drawing_number),
                    ("Scale", meta.sheet_scale),
                    ("Units", meta.default_units),
                    ("Revision", meta.revision),
                ]
                if value
            ]
            if meta_bits:
                st.caption(" · ".join(f"**{label}:** {value}" for label, value in meta_bits))

            box = drawing.bounding_box
            if box:
                st.caption(
                    f"**Bounding box** (fraction of page): "
                    f"x0={box.x0:.3f}, y0={box.y0:.3f}, x1={box.x1:.3f}, y1={box.y1:.3f}"
                    + (f" · confidence: {box.confidence}" if box.confidence else "")
                )
                if box.confidence == "low" or box.notes:
                    st.caption(f"↳ {box.notes or 'Extent flagged low-confidence.'}")
            else:
                st.caption("No bounding box reported for this drawing.")

            st.caption(
                "🔍 Dimensions/datums below came from a cropped, high-resolution re-render of "
                "just this drawing." if drawing.detail_pass_applied else
                "⚠️ No detail pass applied -- results below are from the full-page render only."
            )

            low_confidence_count = sum(
                1 for d in drawing.dimensions if d.confidence == "low"
            ) + sum(1 for d in drawing.elevation_datums if d.confidence == "low")
            if low_confidence_count:
                st.warning(
                    f"{low_confidence_count} reading(s) on this drawing were flagged low-confidence "
                    "-- check the Notes column."
                )

            if drawing.dimensions:
                st.dataframe(
                    [
                        {
                            "ID": dim.dimension_id,
                            "Section/Part": dim.section_part or "",
                            "Element": dim.element or "",
                            "Value": dim.value if dim.value is not None else "",
                            "Unit": dim.unit or "",
                            "Type": dim.type or "",
                            "Orientation": dim.orientation or "",
                            "Start ref": dim.start_reference or "",
                            "End ref": dim.end_reference or "",
                            "Label text": dim.label_text,
                            "Confidence": dim.confidence or "",
                            "Notes": dim.notes or "",
                        }
                        for dim in drawing.dimensions
                    ],
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.caption("No dimensions extracted for this drawing.")

            if drawing.elevation_datums:
                st.markdown("**Elevation datums**")
                st.dataframe(
                    [
                        {
                            "ID": datum.datum_id,
                            "Section/Part": datum.section_part or "",
                            "Level code": datum.level_code or "",
                            "Value": datum.elevation_value if datum.elevation_value is not None else "",
                            "Unit": datum.unit or "",
                            "Label text": datum.label_text,
                            "Confidence": datum.confidence or "",
                            "Notes": datum.notes or "",
                        }
                        for datum in drawing.elevation_datums
                    ],
                    use_container_width=True,
                    hide_index=True,
                )

            if drawing.detail_callouts:
                st.markdown("**Detail callouts**")
                st.dataframe(
                    [
                        {
                            "ID": callout.callout_id,
                            "No.": callout.callout_number or "",
                            "Title": callout.title or "",
                            "Target dwg": callout.target_drawing_number or "",
                            "Section/Part": callout.section_part or "",
                            "Label text": callout.label_text,
                            "Confidence": callout.confidence or "",
                            "Notes": callout.notes or "",
                        }
                        for callout in drawing.detail_callouts
                    ],
                    use_container_width=True,
                    hide_index=True,
                )

            if drawing.quantity_takeoff:
                takeoff_rows = [
                    {
                        "Quantity": field_name,
                        "Value": field.value if field.value is not None else "",
                        "Unit": field.unit or "",
                        "Method": field.method or "",
                        "Confidence": field.confidence or "",
                        "Notes": field.notes or "",
                    }
                    for field_name in drawing.quantity_takeoff.model_fields
                    if (field := getattr(drawing.quantity_takeoff, field_name)) is not None
                ]
                if takeoff_rows:
                    st.markdown("**Quantity takeoff**")
                    st.dataframe(takeoff_rows, use_container_width=True, hide_index=True)

            st.download_button(
                f"Download {drawing.drawing_id} (JSON)",
                data=json.dumps(drawing.model_dump(), indent=2),
                file_name=f"{drawing.drawing_id}.json",
                mime="application/json",
                key=f"{key_prefix}-download-{drawing.drawing_id}",
            )


# --- Sidebar: configuration status -----------------------------------------
with st.sidebar:
    st.header("Configuration")
    gaps = missing_vars()
    all_vars = [
        "AZURE_DOC_INTEL_ENDPOINT",
        "AZURE_DOC_INTEL_KEY",
        "AZURE_FOUNDRY_ENDPOINT",
        "AZURE_FOUNDRY_API_KEY",
    ]
    for var in all_vars:
        if var in gaps:
            st.markdown(f"🔴 `{var}` — not set")
        else:
            st.markdown(f"🟢 `{var}` — set")
    if gaps:
        st.info("Fill in the missing values in your `.env` file (see `.env.example`), then rerun.")

# --- Main: upload + run ------------------------------------------------------
uploaded_files = st.file_uploader(
    "Upload drawing PDF(s)", type=["pdf"], accept_multiple_files=True
)

if uploaded_files:
    run_clicked = st.button("Run extraction", type="primary", disabled=bool(gaps))
    if gaps:
        st.warning("Cannot run extraction until all configuration values above are set.")

    if run_clicked:
        try:
            settings = load_settings()
        except ConfigError as exc:
            st.error(str(exc))
            st.stop()

        results: dict[str, ExtractionResult] = {}
        errors: dict[str, str] = {}

        progress = st.progress(0.0)
        status = st.empty()
        start_time = time.monotonic()
        for i, uploaded in enumerate(uploaded_files):
            elapsed = time.monotonic() - start_time
            status.text(
                f"Processing {i + 1} of {len(uploaded_files)}: {uploaded.name} "
                f"(elapsed {elapsed:.0f}s)"
            )
            try:
                results[uploaded.name] = run_pipeline(
                    uploaded.getvalue(), uploaded.name, settings
                )
            except Exception:
                logging.exception("Pipeline failed for %s", uploaded.name)
                errors[uploaded.name] = (
                    "Extraction failed. Check the app logs/console for details "
                    "(this is usually a credential, endpoint, or model-deployment-name issue)."
                )
            progress.progress((i + 1) / len(uploaded_files))

        status.empty()
        progress.empty()

        st.session_state["results"] = results
        st.session_state["errors"] = errors
        st.session_state["elapsed_seconds"] = time.monotonic() - start_time

# --- Results view -------------------------------------------------------------
results: dict[str, ExtractionResult] = st.session_state.get("results", {})
errors: dict[str, str] = st.session_state.get("errors", {})

if results or errors:
    st.divider()

    elapsed_seconds = st.session_state.get("elapsed_seconds")
    if elapsed_seconds is not None:
        st.caption(f"⏱️ Extraction took {elapsed_seconds:.1f}s")

    if len(results) + len(errors) > 1:
        col1, col2, col3 = st.columns(3)
        col1.metric("Files processed", f"{len(results)}/{len(results) + len(errors)}")
        col2.metric("Drawings found", sum(r.total_drawings for r in results.values()))
        col3.metric(
            "Total dimensions",
            sum(len(d.dimensions) for r in results.values() for d in r.drawings),
        )

    tab_labels = [f"✅ {name}" for name in results] + [f"❌ {name}" for name in errors]
    tabs = st.tabs(tab_labels)

    for tab, filename in zip(tabs[: len(results)], results.keys()):
        with tab:
            render_result(results[filename], key_prefix=filename)

    for tab, filename in zip(tabs[len(results):], errors.keys()):
        with tab:
            st.error(errors[filename])
