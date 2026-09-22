"""Ties the whole pipeline together: PDF bytes in, ExtractionResult out."""
from __future__ import annotations

import logging
import re
from collections import Counter

from pydantic import ValidationError

from . import claude_extractor, doc_intelligence, render
from .doc_intelligence import OcrLine
from .config import Settings
from .schema import (
    BoundingBox,
    DetailCallout,
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


_PLAIN_NUMBER_RE = re.compile(r"^\s*([+-]?\d+(?:\.\d+)?)\s*(?:mm|m|cm|ft|in|%)?\s*$", re.IGNORECASE)

_BARE_CALLOUT_DIGIT_RE = re.compile(r"^\(?\s*(\d{1,2})\s*\)?[.:]?$")


def _ocr_title_callout_number(
    box: BoundingBox, lines: list[OcrLine], search_band: float = 0.07
) -> str | None:
    """Read a drawing's own title-callout digit straight from OCR rather than
    trusting the vision pass for it. On this sheet family the callout circle
    sits at the very bottom-left of a drawing's frame, immediately left of
    its title text -- a short, isolated digit that Document Intelligence's
    OCR reads far more reliably than asking the vision model to transcribe a
    small circled numeral (this is exactly the kind of misread that produced
    two drawings on one sheet both reporting title_callout_number '6').

    Looks for OCR lines that are just a bare 1-2 digit number, sitting near
    the box's bottom edge (title-callout row) and left edge (callout sits
    left of the title text). Returns None -- never guesses -- unless exactly
    one such candidate is found, since the callout circle for a *different*
    drawing/detail-bubble elsewhere in the box must not be picked up.
    """
    band_top = box.y1 - search_band
    candidates = [
        line
        for line in lines
        if _BARE_CALLOUT_DIGIT_RE.match(line.text.strip())
        and line.x0 <= box.x0 + 0.12
        and band_top <= line.y0 <= box.y1 + 0.04
    ]
    if len(candidates) != 1:
        return None
    match = _BARE_CALLOUT_DIGIT_RE.match(candidates[0].text.strip())
    assert match is not None
    return match.group(1)


def _suggest_missing_callout_number(page_drawings: list[Drawing]) -> str | None:
    """When a page's title_callout_numbers look like they should be a
    sequential 1..N set (one per drawing, N = drawing count) but exactly one
    number is used twice and exactly one number in that range is missing,
    that's a strong structural signal for what the miscounted drawing's real
    number should be. Returns that missing number, or None if the pattern
    doesn't hold cleanly -- more than one collision, a missing/non-numeric
    label, or numbers outside the expected 1..N range."""
    numbers = [d.drawing_metadata.title_callout_number for d in page_drawings]
    if not all(n and n.isdigit() for n in numbers):
        return None
    n = len(page_drawings)
    if len(set(numbers)) != n - 1:
        return None  # not exactly one collision
    expected = {str(i) for i in range(1, n + 1)}
    missing = expected - set(numbers)
    if len(missing) == 1 and set(numbers) <= expected:
        return next(iter(missing))
    return None


def _value_from_label_text(label_text: str) -> float | None:
    """Recover a numeric `value` from `label_text` when the model returned
    `value: null` but `label_text` is plainly just a number (optionally with
    a trailing unit), e.g. label_text='1955' or '1955mm'. Deliberately narrow:
    a compound label like '14 RISERS x 280 mm' won't match, since that isn't
    a single measurement -- its total lives in step_formula instead. Purely
    a regex over text the model already returned, so it generalizes to any
    PDF/run rather than patching one drawing's output."""
    match = _PLAIN_NUMBER_RE.match(label_text)
    return float(match.group(1)) if match else None


def _build_dimension(raw_dim: dict, drawing_id: str, dim_index: int, page_number: int) -> Dimension | None:
    step_formula = None
    raw_step = raw_dim.get("step_formula")
    if isinstance(raw_step, dict):
        try:
            step_formula = StepFormula(**raw_step)
        except ValidationError:
            logger.warning("Skipping malformed step_formula on %s: %r", drawing_id, raw_step)

    value = raw_dim.get("value")
    label_text = raw_dim.get("label_text", "")
    if value is None and label_text:
        recovered = _value_from_label_text(label_text)
        if recovered is not None:
            value = recovered

    try:
        return Dimension(
            dimension_id=f"{drawing_id}-DIM{dim_index + 1:02d}",
            orientation=raw_dim.get("orientation"),
            section_part=raw_dim.get("section_part"),
            element=raw_dim.get("element"),
            value=value,
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


def _build_detail_callout(raw_callout: dict, drawing_id: str, callout_index: int, page_number: int) -> DetailCallout | None:
    try:
        return DetailCallout(
            callout_id=f"{drawing_id}-CALLOUT{callout_index + 1:02d}",
            callout_number=raw_callout.get("callout_number"),
            title=raw_callout.get("title"),
            target_drawing_number=raw_callout.get("target_drawing_number"),
            section_part=raw_callout.get("section_part"),
            label_text=raw_callout.get("label_text", ""),
            confidence=raw_callout.get("confidence"),
            notes=raw_callout.get("notes"),
            page_number=page_number,
        )
    except ValidationError:
        logger.warning("Skipping malformed detail_callout on %s: %r", drawing_id, raw_callout)
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


def _clear_title_if_borrowed_from_callout(
    metadata: DrawingMetadata, detail_callouts: list[DetailCallout], drawing_id: str
) -> DrawingMetadata:
    """A drawing's own title must be read from its own title-callout circle,
    never from one of its `detail_callouts` bubbles (a numbered cross-reference
    to a detail on another sheet) -- but that confusion keeps happening on
    hard crops despite prompt wording saying not to. This is the deterministic
    half of that fix: if `drawing_title` is character-for-character identical
    (case-insensitive) to one of this same drawing's own detail_callouts
    titles, it was almost certainly copied from that bubble rather than read
    from the drawing's real title block, so clear it rather than keep a value
    we can prove is wrong. Purely structural -- compares two lists of data the
    model already returned, with no hardcoded titles -- so it generalizes to
    any sheet/project.
    """
    title = metadata.drawing_title
    if not title:
        return metadata
    title_norm = title.strip().lower()

    for callout in detail_callouts:
        callout_title = callout.title
        if not callout_title or callout_title.strip().lower() in ("", "unclear"):
            continue
        if callout_title.strip().lower() == title_norm:
            logger.warning(
                "%s: drawing_title %r is identical to detail_callout %s's title -- "
                "likely borrowed from that callout bubble instead of this drawing's own "
                "title block; clearing drawing_title/title_callout_number",
                drawing_id, title, callout.callout_id,
            )
            return metadata.model_copy(update={"drawing_title": None, "title_callout_number": None})

    return metadata


_CROSS_DRAWING_LEAK_RE = re.compile(
    r"next drawing|neighboring drawing|different drawing|cut off|extends beyond|"
    r"not fully captured|separate sheet region|outside (?:this|the) drawing|"
    r"outside (?:this|the) (?:page|crop)'s visible area",
    re.IGNORECASE,
)


def _looks_like_cross_drawing_leak(callout: DetailCallout) -> bool:
    """True if this callout's own notes/section_part admit it's describing
    content that belongs to a different drawing rather than this one -- a
    known detail-pass crop-leakage pattern where a sliver of a neighboring
    drawing's frame/title sneaks into this drawing's crop, and the model
    says so itself (e.g. '...this next drawing... is cut off... extends
    beyond the page's visible area'). Purely a text pattern over what the
    model already wrote in its own notes, so it generalizes to any
    drawing/run rather than special-casing one entry."""
    text = f"{callout.notes or ''} {callout.section_part or ''}"
    return bool(_CROSS_DRAWING_LEAK_RE.search(text))


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

    detail_callouts = []
    for callout_index, raw_callout in enumerate(raw_drawing.get("detail_callouts", [])):
        callout = _build_detail_callout(raw_callout, drawing_id, callout_index, page_number)
        if callout is None:
            continue
        if _looks_like_cross_drawing_leak(callout):
            logger.warning(
                "%s: dropping detail_callout %r -- its own notes/section_part admit it "
                "describes content outside this drawing (detail-pass crop leakage): %r",
                drawing_id, callout.label_text, callout.notes,
            )
            continue
        detail_callouts.append(callout)

    raw_metadata = raw_drawing.get("drawing_metadata") or {}
    try:
        metadata = DrawingMetadata(**raw_metadata) if isinstance(raw_metadata, dict) else DrawingMetadata()
    except ValidationError:
        logger.warning("Skipping malformed drawing_metadata on %s: %r", drawing_id, raw_metadata)
        metadata = DrawingMetadata()

    metadata = _clear_title_if_borrowed_from_callout(metadata, detail_callouts, drawing_id)

    quantity_takeoff = _build_quantity_takeoff(raw_drawing.get("quantity_takeoff"), drawing_id)

    return {
        "drawing_type": raw_drawing.get("drawing_type", "other"),
        "drawing_metadata": metadata,
        "dimensions": dimensions,
        "elevation_datums": elevation_datums,
        "detail_callouts": detail_callouts,
        "quantity_takeoff": quantity_takeoff,
    }


def _ocr_text_for_box(
    lines: list[OcrLine],
    box: BoundingBox,
    sibling_boxes: list[BoundingBox],
    pad: float = 0.015,
) -> str:
    """Scope OCR text down to just the lines belonging to this drawing,
    instead of handing the detail pass the whole page's OCR text.

    A small pad picks up text sitting right at this drawing's own edge, but
    on a sheet of stacked/touching drawings (zero gap between boxes -- the
    common case here) that same pad reaches straight into the neighboring
    drawing, exactly where ITS title sits (title text conventionally sits at
    the bottom edge of each drawing's frame). That's what was pulling a
    neighbor's title into this drawing's detail-pass context even after the
    OCR was scoped down at all -- so any line whose center falls inside a
    sibling drawing's own (unpadded) box is excluded here, regardless of
    whether this box's padding also reaches it.
    """
    x0, y0 = box.x0 - pad, box.y0 - pad
    x1, y1 = box.x1 + pad, box.y1 + pad
    matched = []
    for line in lines:
        center_x, center_y = (line.x0 + line.x1) / 2, (line.y0 + line.y1) / 2
        if any(
            sb.x0 <= center_x <= sb.x1 and sb.y0 <= center_y <= sb.y1
            for sb in sibling_boxes
        ):
            continue
        if line.x1 >= x0 and line.x0 <= x1 and line.y1 >= y0 and line.y0 <= y1:
            matched.append(line.text)
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
    merged["detail_callouts"] = [d for raw in raw_detail_drawings for d in raw.get("detail_callouts", [])]
    return merged


def _find_drawing_directly_above(
    drawing_id: str, box: BoundingBox, page_boxes: dict[str, BoundingBox]
) -> str | None:
    """Find the id of the drawing on the same page whose box sits immediately
    above `box` in the same column: their x-ranges overlap, its bottom edge
    is at or above this box's top edge, and it's the closest such neighbor.
    Purely geometric (uses only bounding_box coordinates already produced),
    so it applies to any sheet layout, not just this one."""
    best_id = None
    best_gap = None
    for other_id, other_box in page_boxes.items():
        if other_id == drawing_id:
            continue
        x_overlap = min(box.x1, other_box.x1) - max(box.x0, other_box.x0)
        if x_overlap <= 0:
            continue
        gap = box.y0 - other_box.y1
        if gap < -1e-6:
            continue
        if best_gap is None or gap < best_gap:
            best_gap = gap
            best_id = other_id
    return best_id


_TITLE_LIKE_RE = re.compile(r"[A-Za-z]{3,}")


def _looks_like_title_text(text: str) -> bool:
    """Heuristic for 'this OCR line is descriptive title-block text, not a
    bare dimension number or a lone grid-bubble digit' -- at least one run of
    3+ letters, and not purely numeric. Deliberately generic (no drawing- or
    project-specific words) so it doesn't just pattern-match this one sheet's
    vocabulary; it's only used to decide whether a line found just below a
    box's reported bottom edge is worth pulling that edge down to enclose,
    not to identify what the title actually says."""
    text = text.strip()
    return bool(text) and not text.isdigit() and bool(_TITLE_LIKE_RE.search(text))


def _find_touching_neighbor_below(
    drawing_id: str, box: BoundingBox, boxes: dict[str, BoundingBox], tol: float = 0.01
) -> str | None:
    """The geometric inverse of _find_drawing_directly_above: the closest
    box in the same column (x-ranges overlap) whose top edge sits at or just
    below this box's bottom edge."""
    best_id, best_gap = None, None
    for other_id, other_box in boxes.items():
        if other_id == drawing_id:
            continue
        x_overlap = min(box.x1, other_box.x1) - max(box.x0, other_box.x0)
        if x_overlap <= 0:
            continue
        gap = other_box.y0 - box.y1
        if gap < -tol:
            continue
        if best_gap is None or gap < best_gap:
            best_gap, best_id = gap, other_id
    return best_id


def _snap_boxes_to_include_own_title(
    boxes: dict[str, BoundingBox], lines: list[OcrLine], search_margin: float = 0.10
) -> dict[str, BoundingBox]:
    """Confirmed empirically (by tracing real OCR line coordinates against
    real bounding boxes): Pass 1 sometimes draws a drawing's bottom edge just
    above its own title-callout line -- the gap between the box edge and
    the title varies from ~0 to ~7% of page height depending on the sheet,
    so no fixed offset can fix this. That gap then causes the render_crop
    for the box UNDERNEATH to include the neighbor's title instead of its
    own, which is what was producing the observed 'this drawing has the
    title that rightfully belongs to the one above it' pattern.

    For each box, this looks for a title-like OCR line sitting just below
    its reported bottom edge, with overlapping x-range (same column, so a
    line from an unrelated adjacent column can't be pulled in), and expands
    the box to enclose it. Whatever box was touching it below gets its top
    edge pushed down to match, so the two don't end up overlapping and the
    same title doesn't end up in both. Purely geometric plus the generic
    "is this descriptive text" heuristic above -- no drawing-specific
    keywords or magic offsets -- so it applies to any sheet layout.
    """
    adjusted = dict(boxes)

    for drawing_id, box in boxes.items():
        new_y1 = box.y1
        for line in lines:
            if line.y1 <= box.y1 + 1e-6:
                continue  # already fully enclosed by this box -- nothing to fix
            if line.y0 > box.y1 + search_margin:
                continue  # starts too far below to plausibly be this box's own title
            x_overlap = min(box.x1, line.x1) - max(box.x0, line.x0)
            if x_overlap <= 0:
                continue  # different column
            if not _looks_like_title_text(line.text):
                continue
            new_y1 = max(new_y1, min(1.0, line.y1))

        if new_y1 <= box.y1:
            continue

        logger.warning(
            "%s: expanding bounding_box.y1 from %.4f to %.4f to enclose a title-like "
            "line found just below its reported edge (was about to spill into whatever "
            "box sits underneath)",
            drawing_id, box.y1, new_y1,
        )
        # Base the update on the current entry in `adjusted`, not the original
        # `box` -- a still-earlier iteration (the box above this one) may
        # already have pushed this box's own y0 down, and copying from the
        # stale `box` here would silently discard that.
        adjusted[drawing_id] = adjusted[drawing_id].model_copy(update={"y1": new_y1})

        below_id = _find_touching_neighbor_below(drawing_id, box, boxes)
        if below_id is not None:
            below_box = adjusted[below_id]
            if new_y1 > below_box.y0 and new_y1 < below_box.y1:
                adjusted[below_id] = below_box.model_copy(update={"y0": new_y1})

    return adjusted


def _clamp_overlapping_boxes(boxes: dict[str, BoundingBox]) -> dict[str, BoundingBox]:
    """Final safety pass after all other box adjustments: no two drawings'
    boxes in the same column should vertically overlap, since the detail
    pass renders each box's exact coordinates with no further trimming --
    any leftover overlap here (from a model-reported box generously padded
    per the extraction prompt, not from the intentional title-inclusion
    expansion above) would let one drawing's high-res crop bleed into its
    neighbor's frame/title, which is what has produced spurious
    detail_callouts describing a neighboring drawing's content. For each
    overlapping pair in the same column, clips the box that starts higher
    to stop exactly where the other begins, rather than leaving the two
    crops overlapping. Purely geometric (x/y overlap only, no OCR/text
    involved), so it applies to any sheet layout, not just stacked columns.
    """
    adjusted = dict(boxes)
    ids = list(adjusted)
    for i, id_a in enumerate(ids):
        for id_b in ids[i + 1:]:
            a, b = adjusted[id_a], adjusted[id_b]
            x_overlap = min(a.x1, b.x1) - max(a.x0, b.x0)
            if x_overlap <= 0:
                continue
            upper_id, lower_id = (id_a, id_b) if a.y0 <= b.y0 else (id_b, id_a)
            upper, lower = adjusted[upper_id], adjusted[lower_id]
            if upper.y1 > lower.y0:
                logger.warning(
                    "%s and %s overlap after box adjustments (%.4f > %.4f) -- clipping "
                    "%s's bottom edge to stop where %s begins, so their crops don't bleed "
                    "into each other",
                    upper_id, lower_id, upper.y1, lower.y0, upper_id, lower_id,
                )
                adjusted[upper_id] = upper.model_copy(update={"y1": lower.y0})
    return adjusted


def _normalize_callout_key(text: str | None) -> str | None:
    """Collapse a detail-callout's title (or, failing that, its raw
    label_text) down to its bare alphabetic content -- letters only,
    uppercased, digits/punctuation/whitespace stripped -- so the same
    recurring sheet-wide annotation can be recognized across drawings even
    when one drawing's read dropped part of the title (e.g. 'STEEL HANDRAIL
    DETAIL-1-5' vs just 'DETAIL-1-5' vs 'unclear'). Returns None when there's
    not enough legible text to key on."""
    if not text or text.strip().lower() == "unclear":
        return None
    letters = re.sub(r"[^A-Za-z]", "", text).upper()
    return letters if len(letters) >= 4 else None


def _reconcile_detail_callouts(page_drawings: list[Drawing]) -> None:
    """Cross-check each drawing's detail_callouts against how the SAME
    annotation was read on other drawings on this page. A numbered detail
    bubble (e.g. '6 STEEL HANDRAIL DETAIL-1-5' / 'XXX-DWG-AR-AR-510102') is
    printed identically on every drawing on a sheet, so different readings of
    it across drawings are misreads, not real variation -- but a plain vote
    count is the wrong way to pick the correct one: the same misread (e.g.
    several drawings all dropping "STEEL HANDRAIL" down to just "DETAIL-1-5")
    can easily outnumber the one drawing that read the full text correctly.
    Instead this scores each distinct reading by how COMPLETE its title is
    (a longer, non-truncated title is strictly more informative than a short
    fragment, however many drawings share that fragment) and only falls back
    to vote count to break ties between equally-complete readings. Requires
    at least 3 legible readings before acting, so a genuine one-off callout
    (seen on only one or two drawings) is never touched. Also dedupes any
    drawing left with two callouts now sharing identical content.
    """
    groups: dict[str, list[DetailCallout]] = {}
    for d in page_drawings:
        for callout in d.detail_callouts:
            key = _normalize_callout_key(callout.title) or _normalize_callout_key(callout.label_text)
            if key is None:
                continue
            # Merge into an existing group if one key contains the other
            # (covers a title read as only a fragment of the full text).
            matched_key = next((k for k in groups if k in key or key in k), key)
            groups.setdefault(matched_key, []).append(callout)

    for callouts in groups.values():
        if len(callouts) < 3:
            continue
        candidates = [
            c for c in callouts
            if c.callout_number and c.callout_number != "unclear"
            and c.target_drawing_number and c.target_drawing_number != "unclear"
        ]
        if not candidates:
            continue
        counts = Counter((c.callout_number, c.title, c.target_drawing_number) for c in candidates)

        def _completeness(c: DetailCallout) -> tuple[int, int]:
            title_len = len(c.title.strip()) if c.title and c.title.strip().lower() != "unclear" else 0
            return (title_len, counts[(c.callout_number, c.title, c.target_drawing_number)])

        canonical = max(candidates, key=_completeness)
        number, title, target = canonical.callout_number, canonical.title, canonical.target_drawing_number
        for c in callouts:
            if (c.callout_number, c.title, c.target_drawing_number) == (number, title, target):
                continue
            logger.warning(
                "%s: detail_callout %r read as (number=%r, title=%r, target=%r) but the most "
                "complete reading on this page is (number=%r, title=%r, target=%r) -- overriding",
                c.callout_id, c.label_text, c.callout_number, c.title, c.target_drawing_number,
                number, title, target,
            )
            c.callout_number = number
            c.title = title
            c.target_drawing_number = target

    for d in page_drawings:
        seen: set[tuple[str | None, str | None, str | None]] = set()
        deduped = []
        for c in d.detail_callouts:
            sig = (c.callout_number, c.title, c.target_drawing_number)
            if sig in seen:
                continue
            seen.add(sig)
            deduped.append(c)
        if len(deduped) != len(d.detail_callouts):
            d.detail_callouts = deduped


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
        seen_callout_numbers: dict[str, str] = {}
        page_boxes = {d.drawing_id: d.bounding_box for d in page_drawings if d.bounding_box is not None}

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

            # title_callout_number is a short, low-ambiguity signal (typically
            # sequential 1..N across the sheet) -- a collision here is a
            # stronger sign of mislabeling than a duplicate title string,
            # which can vary in phrasing even when correct.
            callout_number = d.drawing_metadata.title_callout_number
            if callout_number:
                if callout_number in seen_callout_numbers:
                    suggestion = _suggest_missing_callout_number(page_drawings)
                    suggestion_note = (
                        f" -- the only number missing from this page's 1..{len(page_drawings)} "
                        f"sequence is {suggestion!r}, likely the real value for one of them"
                        if suggestion else ""
                    )
                    logger.warning(
                        "Page %s: %s and %s report the same title_callout_number %r%s",
                        page_number, seen_callout_numbers[callout_number], d.drawing_id,
                        callout_number, suggestion_note,
                    )
                else:
                    seen_callout_numbers[callout_number] = d.drawing_id

            if not d.dimensions and not d.elevation_datums:
                logger.warning(
                    "%s has no dimensions or elevation_datums -- possibly a title "
                    "block/legend/keyplan region mistakenly segmented as a drawing, "
                    "or a detail pass that found nothing",
                    d.drawing_id,
                )

        # Positional check: on a page of stacked/columned drawings, a
        # recurring failure is a drawing reporting the title_callout_number
        # that rightfully belongs to whichever drawing sits immediately
        # above it (same column, closest box above) -- as if the model's
        # attention lagged one drawing behind while reading down a stack.
        # This looks only at geometry (bounding_box) plus the number each
        # drawing already reported, so it's structural and sheet-agnostic.
        callout_by_id = {d.drawing_id: d.drawing_metadata.title_callout_number for d in page_drawings}
        for d in page_drawings:
            if d.bounding_box is None or not d.drawing_metadata.title_callout_number:
                continue
            above_id = _find_drawing_directly_above(d.drawing_id, d.bounding_box, page_boxes)
            if above_id and callout_by_id.get(above_id) == d.drawing_metadata.title_callout_number:
                logger.warning(
                    "Page %s: %s reports title_callout_number %r, the same as %s directly "
                    "above it -- likely reused the drawing above's title instead of its own",
                    page_number, d.drawing_id, d.drawing_metadata.title_callout_number, above_id,
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
        page_start = len(drawings)
        page_ocr_text = ocr_text.get(page_number, "")
        page_ocr_lines = ocr_lines.get(page_number, [])
        raw_drawings = claude_extractor.extract_page(
            image_bytes=image_bytes,
            ocr_text=page_ocr_text,
            page_number=page_number,
            settings=settings,
        )

        # Bounding boxes for every drawing on this page, computed up front so
        # each drawing's detail pass knows where its siblings sit -- needed
        # to keep OCR context (and title text especially) from leaking across
        # touching/stacked drawing boundaries.
        page_drawing_ids = [f"P{page_number}-D{i + 1}" for i in range(len(raw_drawings))]
        page_bounding_boxes = [
            _build_bounding_box(raw_drawing.get("bounding_box"), page_drawing_ids[i])
            for i, raw_drawing in enumerate(raw_drawings)
        ]

        # Snap each box to fully enclose its own title-callout line rather
        # than cutting it off at a boundary Pass 1 drew slightly too early
        # (see _snap_boxes_to_include_own_title) -- this is what was causing
        # a drawing's detail-pass crop to show its neighbor's title instead
        # of its own.
        boxes_by_id = {
            page_drawing_ids[i]: b for i, b in enumerate(page_bounding_boxes) if b is not None
        }
        if boxes_by_id and page_ocr_lines:
            boxes_by_id = _snap_boxes_to_include_own_title(boxes_by_id, page_ocr_lines)
        if boxes_by_id:
            boxes_by_id = _clamp_overlapping_boxes(boxes_by_id)
        page_bounding_boxes = [boxes_by_id.get(drawing_id) for drawing_id in page_drawing_ids]

        for drawing_index, raw_drawing in enumerate(raw_drawings):
            drawing_id = page_drawing_ids[drawing_index]

            fields = _build_drawing_fields(raw_drawing, drawing_id, page_number)
            bounding_box = page_bounding_boxes[drawing_index]
            detail_pass_applied = False

            # 4. Detail pass: re-render just this drawing's bounding box at a
            #    much higher effective DPI and re-extract from that crop
            #    alone, so small dimension text that was unreadable in the
            #    full-page pass gets another, much sharper look.
            if bounding_box is not None:
                sibling_boxes = [
                    b for i, b in enumerate(page_bounding_boxes)
                    if i != drawing_index and b is not None
                ]
                scoped_ocr_text = (
                    _ocr_text_for_box(page_ocr_lines, bounding_box, sibling_boxes)
                    if page_ocr_lines else page_ocr_text
                )
                raw_detail = _run_detail_pass(
                    pdf_bytes, page_number, bounding_box, scoped_ocr_text, drawing_id, settings
                )
                if raw_detail is not None:
                    fields = _build_drawing_fields(raw_detail, drawing_id, page_number)
                    detail_pass_applied = True

            # 5. Cross-check the title-callout number against OCR: a short,
            #    isolated digit is exactly what Document Intelligence reads
            #    most reliably, and is a cheap, deterministic correction for
            #    a class of vision misread that has shown up in practice
            #    (see _ocr_title_callout_number).
            if bounding_box is not None and page_ocr_lines:
                ocr_number = _ocr_title_callout_number(bounding_box, page_ocr_lines)
                metadata = fields["drawing_metadata"]
                if ocr_number is not None and ocr_number != metadata.title_callout_number:
                    logger.warning(
                        "%s: OCR reads title-callout number %r but the vision pass "
                        "returned %r -- overriding with the OCR reading",
                        drawing_id, ocr_number, metadata.title_callout_number,
                    )
                    fields["drawing_metadata"] = metadata.model_copy(
                        update={"title_callout_number": ocr_number}
                    )

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

        # 6. Reconcile detail_callouts across this page's drawings: the same
        #    numbered detail bubble is printed identically on every drawing on
        #    a sheet, so if most drawings agree on one and a minority disagree,
        #    the minority almost certainly misread it (see _reconcile_detail_callouts).
        _reconcile_detail_callouts(drawings[page_start:])

    _warn_on_suspect_drawings(drawings)

    return ExtractionResult(
        source_file=filename,
        total_pages=total_pages,
        total_drawings=len(drawings),
        drawings=drawings,
    )
