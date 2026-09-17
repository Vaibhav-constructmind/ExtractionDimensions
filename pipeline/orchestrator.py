"""Ties the whole pipeline together: PDF bytes in, ExtractionResult out."""
from __future__ import annotations

import logging

from pydantic import ValidationError

from . import claude_extractor, doc_intelligence, render
from .doc_intelligence import OcrLine
from .config import Settings
from .schema import (
    BoundingBox,
    Dimension,
    Drawing,
    DrawingMetadata,
    ElevationDatum,
    ExtractionResult,
    QuantityField,
    QuantityTakeoff,
    StepFormula,
)

logger = logging.getLogger(__name__)


def _build_dimension(raw_dim: dict, drawing_id: str, dim_index: int, page_number: int) -> Dimension | None:
    step_formula = None
    raw_step = raw_dim.get("step_formula")
    if isinstance(raw_step, dict):
        try:
            step_formula = StepFormula(**raw_step)
        except ValidationError:
            logger.warning("Skipping malformed step_formula on %s: %r", drawing_id, raw_step)

    try:
        return Dimension(
            dimension_id=f"{drawing_id}-DIM{dim_index + 1:02d}",
            orientation=raw_dim.get("orientation"),
            section_part=raw_dim.get("section_part"),
            element=raw_dim.get("element"),
            value=raw_dim.get("value"),
            unit=raw_dim.get("unit"),
            start_reference=raw_dim.get("start_reference"),
            end_reference=raw_dim.get("end_reference"),
            label_text=raw_dim.get("label_text", ""),
            type=raw_dim.get("type"),
            step_formula=step_formula,
            confidence=raw_dim.get("confidence"),
            notes=raw_dim.get("notes"),
            page_number=page_number,
        )
    except ValidationError:
        # A single malformed dimension shouldn't take down the whole run.
        logger.warning("Skipping malformed dimension on %s: %r", drawing_id, raw_dim)
        return None


def _build_elevation_datum(raw_datum: dict, drawing_id: str, datum_index: int, page_number: int) -> ElevationDatum | None:
    try:
        return ElevationDatum(
            datum_id=f"{drawing_id}-DATUM{datum_index + 1:02d}",
            section_part=raw_datum.get("section_part"),
            level_code=raw_datum.get("level_code"),
            elevation_value=raw_datum.get("elevation_value"),
            unit=raw_datum.get("unit"),
            label_text=raw_datum.get("label_text", ""),
            confidence=raw_datum.get("confidence"),
            notes=raw_datum.get("notes"),
            page_number=page_number,
        )
    except ValidationError:
        logger.warning("Skipping malformed elevation datum on %s: %r", drawing_id, raw_datum)
        return None


def _build_quantity_field(raw_field: object) -> QuantityField | None:
    if not isinstance(raw_field, dict):
        return None
    try:
        return QuantityField(
            value=raw_field.get("value"),
            unit=raw_field.get("unit"),
            method=raw_field.get("method"),
            confidence=raw_field.get("confidence"),
            notes=raw_field.get("notes"),
        )
    except ValidationError:
        return None


def _build_quantity_takeoff(raw_takeoff: object, drawing_id: str) -> QuantityTakeoff | None:
    if raw_takeoff is None:
        return None
    if not isinstance(raw_takeoff, dict):
        logger.warning("Skipping malformed quantity_takeoff on %s: %r", drawing_id, raw_takeoff)
        return None
    try:
        return QuantityTakeoff(**{
            field_name: _build_quantity_field(raw_takeoff.get(field_name))
            for field_name in QuantityTakeoff.model_fields
        })
    except ValidationError:
        logger.warning("Skipping malformed quantity_takeoff on %s: %r", drawing_id, raw_takeoff)
        return None


def _build_bounding_box(raw_box: object, drawing_id: str) -> BoundingBox | None:
    if raw_box is None:
        return None
    if not isinstance(raw_box, dict):
        logger.warning("Skipping malformed bounding_box on %s: %r", drawing_id, raw_box)
        return None

    try:
        x0, y0, x1, y1 = (float(raw_box[k]) for k in ("x0", "y0", "x1", "y1"))
    except (KeyError, TypeError, ValueError):
        logger.warning("Skipping malformed bounding_box on %s: %r", drawing_id, raw_box)
        return None

    # Claude was told to pad generously, so clamp rather than reject -- a box
    # that strays slightly outside [0,1] or has swapped edges is still usable.
    x0, x1 = sorted((max(0.0, min(1.0, x0)), max(0.0, min(1.0, x1))))
    y0, y1 = sorted((max(0.0, min(1.0, y0)), max(0.0, min(1.0, y1))))

    try:
        return BoundingBox(
            x0=x0, y0=y0, x1=x1, y1=y1,
            confidence=raw_box.get("confidence"),
            notes=raw_box.get("notes"),
        )
    except ValidationError:
        logger.warning("Skipping malformed bounding_box on %s: %r", drawing_id, raw_box)
        return None


def _build_drawing_fields(raw_drawing: dict, drawing_id: str, page_number: int) -> dict:
    """Build the dimensions/elevation_datums/metadata/quantity_takeoff/drawing_type
    for one drawing out of one raw Claude tool-call entry. Shared between the
    full-page pass and the per-drawing high-res detail pass so both build
    these fields identically."""
    dimensions = [
        dim
        for dim_index, raw_dim in enumerate(raw_drawing.get("dimensions", []))
        if (dim := _build_dimension(raw_dim, drawing_id, dim_index, page_number)) is not None
    ]

    elevation_datums = [
        datum
        for datum_index, raw_datum in enumerate(raw_drawing.get("elevation_datums", []))
        if (datum := _build_elevation_datum(raw_datum, drawing_id, datum_index, page_number)) is not None
    ]

    raw_metadata = raw_drawing.get("drawing_metadata") or {}
    try:
        metadata = DrawingMetadata(**raw_metadata) if isinstance(raw_metadata, dict) else DrawingMetadata()
    except ValidationError:
        logger.warning("Skipping malformed drawing_metadata on %s: %r", drawing_id, raw_metadata)
        metadata = DrawingMetadata()

    quantity_takeoff = _build_quantity_takeoff(raw_drawing.get("quantity_takeoff"), drawing_id)

    return {
        "drawing_type": raw_drawing.get("drawing_type", "other"),
        "drawing_metadata": metadata,
        "dimensions": dimensions,
        "elevation_datums": elevation_datums,
        "quantity_takeoff": quantity_takeoff,
    }


def _ocr_text_for_box(lines: list[OcrLine], box: BoundingBox, pad: float = 0.03) -> str:
    """Scope OCR text down to just the lines overlapping this drawing's
    (slightly padded) bounding box, instead of handing the detail pass the
    whole page's OCR text.

    Without this, every drawing's detail-pass call could see every other
    drawing's title-block text too -- on a sheet packed with several similar
    stair-plan drawings, that's exactly the kind of ambiguity that lets a
    model pull the wrong title (or other text) off a neighboring drawing
    even when it read the crop's own dimensions correctly.
    """
    x0, y0 = box.x0 - pad, box.y0 - pad
    x1, y1 = box.x1 + pad, box.y1 + pad
    matched = [
        line.text
        for line in lines
        if line.x1 >= x0 and line.x0 <= x1 and line.y1 >= y0 and line.y0 <= y1
    ]
    return "\n".join(matched)


def _run_detail_pass(
    pdf_bytes: bytes,
    page_number: int,
    bounding_box: BoundingBox,
    ocr_text: str,
    drawing_id: str,
    settings: Settings,
) -> dict | None:
    """Crop this drawing's bounding box out of the page, re-render it alone at
    a much higher effective DPI, and re-extract from that crop alone.

    Returns the merged raw-drawing dict to rebuild this Drawing's fields from,
    or None if the detail pass failed or found nothing (caller should keep
    the original full-page-pass results in that case).
    """
    try:
        crop_bytes = render.render_crop(
            pdf_bytes,
            page_number,
            bounding_box.x0,
            bounding_box.y0,
            bounding_box.x1,
            bounding_box.y1,
            max_dpi=settings.detail_render_dpi,
        )
    except Exception:
        logger.warning("Detail-pass crop render failed for %s; keeping full-page-pass results", drawing_id, exc_info=True)
        return None

    try:
        raw_detail_drawings = claude_extractor.extract_page(
            image_bytes=crop_bytes,
            ocr_text=ocr_text,
            page_number=page_number,
            settings=settings,
        )
    except Exception:
        logger.warning("Detail-pass extraction failed for %s; keeping full-page-pass results", drawing_id, exc_info=True)
        return None

    if not raw_detail_drawings:
        logger.warning("Detail pass for %s found no drawings in its crop; keeping full-page-pass results", drawing_id)
        return None

    if len(raw_detail_drawings) == 1:
        return raw_detail_drawings[0]

    # The crop was meant to hold exactly this one drawing; if Claude still
    # split it, treat that as an over-segmentation of the same drawing and
    # merge everything back into one (pooling all found dimensions/datums)
    # rather than silently dropping any of them.
    logger.warning(
        "Detail-pass crop for %s still split into %d sub-drawings; merging them back into one",
        drawing_id, len(raw_detail_drawings),
    )
    merged = dict(raw_detail_drawings[0])
    merged["dimensions"] = [d for raw in raw_detail_drawings for d in raw.get("dimensions", [])]
    merged["elevation_datums"] = [d for raw in raw_detail_drawings for d in raw.get("elevation_datums", [])]
    return merged


def _warn_on_suspect_drawings(drawings: list[Drawing]) -> None:
    """Best-effort consistency check over the finished result: flag drawings
    on the same page that share a title, and drawings with no dimensions or
    datums at all. Both patterns have shown up as real extraction mistakes --
    a drawing mislabeled with a neighboring drawing's title, or a sheet's
    title block/legend/keyplan region mistakenly segmented as its own empty
    'drawing' -- so surface them for a manual check rather than staying
    silent about it."""
    by_page: dict[int, list[Drawing]] = {}
    for d in drawings:
        by_page.setdefault(d.page_number, []).append(d)

    for page_number, page_drawings in by_page.items():
        seen_titles: dict[str, str] = {}
        for d in page_drawings:
            title = d.drawing_metadata.drawing_title
            if title:
                if title in seen_titles:
                    logger.warning(
                        "Page %s: %s and %s share the same drawing_title %r -- "
                        "possible mislabeling, worth a manual check",
                        page_number, seen_titles[title], d.drawing_id, title,
                    )
                else:
                    seen_titles[title] = d.drawing_id

            if not d.dimensions and not d.elevation_datums:
                logger.warning(
                    "%s has no dimensions or elevation_datums -- possibly a title "
                    "block/legend/keyplan region mistakenly segmented as a drawing, "
                    "or a detail pass that found nothing",
                    d.drawing_id,
                )


def run_pipeline(pdf_bytes: bytes, filename: str, settings: Settings) -> ExtractionResult:
    # 1. OCR / layout via Document Intelligence.
    analysis = doc_intelligence.analyze_pdf(pdf_bytes, settings)
    ocr_text = doc_intelligence.text_by_page(analysis)
    ocr_lines = doc_intelligence.lines_by_page(analysis)
    total_pages = doc_intelligence.page_count(analysis)

    # 2. Render each page to an image for vision input.
    page_images = render.render_pages(pdf_bytes, dpi=settings.render_dpi)

    drawings: list[Drawing] = []

    # 3. Per page: ask Claude to segment drawings and extract dimensions
    #    (a low-res pass over the whole page, just to find each drawing and
    #    its bounding box).
    for index, image_bytes in enumerate(page_images):
        page_number = index + 1
        page_ocr_text = ocr_text.get(page_number, "")
        page_ocr_lines = ocr_lines.get(page_number, [])
        raw_drawings = claude_extractor.extract_page(
            image_bytes=image_bytes,
            ocr_text=page_ocr_text,
            page_number=page_number,
            settings=settings,
        )

        for drawing_index, raw_drawing in enumerate(raw_drawings):
            drawing_id = f"P{page_number}-D{drawing_index + 1}"

            fields = _build_drawing_fields(raw_drawing, drawing_id, page_number)
            bounding_box = _build_bounding_box(raw_drawing.get("bounding_box"), drawing_id)
            detail_pass_applied = False

            # 4. Detail pass: re-render just this drawing's bounding box at a
            #    much higher effective DPI and re-extract from that crop
            #    alone, so small dimension text that was unreadable in the
            #    full-page pass gets another, much sharper look.
            if bounding_box is not None:
                scoped_ocr_text = _ocr_text_for_box(page_ocr_lines, bounding_box) if page_ocr_lines else page_ocr_text
                raw_detail = _run_detail_pass(
                    pdf_bytes, page_number, bounding_box, scoped_ocr_text, drawing_id, settings
                )
                if raw_detail is not None:
                    fields = _build_drawing_fields(raw_detail, drawing_id, page_number)
                    detail_pass_applied = True

            try:
                drawings.append(
                    Drawing(
                        drawing_id=drawing_id,
                        page_number=page_number,
                        bounding_box=bounding_box,
                        detail_pass_applied=detail_pass_applied,
                        **fields,
                    )
                )
            except ValidationError:
                logger.warning("Skipping malformed drawing %s: %r", drawing_id, raw_drawing)

    _warn_on_suspect_drawings(drawings)

    return ExtractionResult(
        source_file=filename,
        total_pages=total_pages,
        total_drawings=len(drawings),
        drawings=drawings,
    )
