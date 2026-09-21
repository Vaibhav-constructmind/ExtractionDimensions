"""Claude Sonnet-5 (Azure AI Foundry) vision extraction.

Sends each rendered page image, plus the Document Intelligence OCR text for
that page, to Claude and forces a structured tool call that records every
distinct drawing on the page and every dimension found in each drawing.

Azure AI Foundry Claude deployments expose an Anthropic Messages-compatible
API, so this uses the official `anthropic` SDK with `base_url` pointed at the
Foundry endpoint. If your specific Foundry deployment uses a different wire
format, this is the one place that would need adjusting.
"""
from __future__ import annotations

import base64
import logging

from anthropic import Anthropic

from .config import Settings

logger = logging.getLogger(__name__)

DRAWING_TYPES = ["elevation", "section", "isometric", "plan", "detail", "schedule", "other"]
ORIENTATIONS = ["vertical", "horizontal", "diagonal", "other"]
DIMENSION_TYPES = [
    "clearance", "headroom", "floor_to_floor", "level_difference", "guardrail_height",
    "stair_rise", "tread_going", "wall_thickness", "width", "other",
]
CONFIDENCE_LEVELS = ["high", "medium", "low"]

TOOL_NAME = "record_drawings"

_DIMENSION_ITEM_SCHEMA = {
    "type": "object",
    "properties": {
        "orientation": {"type": "string", "enum": ORIENTATIONS},
        "section_part": {
            "type": "string",
            "description": "Where on the drawing this is, e.g. 'Shaft / Vent Alcove (Upper-Left)'.",
        },
        "element": {
            "type": "string",
            "description": "What is being dimensioned, e.g. 'Ventilation intake shaft clear height'.",
        },
        "value": {
            "type": "number",
            "description": "Numeric value in `unit`, e.g. 5550. Omit if not a clean number.",
        },
        "unit": {"type": "string", "description": "e.g. mm, m, ft, in"},
        "start_reference": {
            "type": "string",
            "description": "Where the dimension line starts, e.g. 'Top of Slab (+601.88 T.O.S.)'.",
        },
        "end_reference": {
            "type": "string",
            "description": "Where the dimension line ends, e.g. 'Underside of top concrete roof slab'.",
        },
        "label_text": {
            "type": "string",
            "description": (
                "Raw text exactly as printed on the drawing, e.g. '5550' or '15 RISERS @ 175 = 2625'. "
                "Use 'unclear' if illegible."
            ),
        },
        "type": {"type": "string", "enum": DIMENSION_TYPES},
        "step_formula": {
            "type": "object",
            "description": "Only for stair_rise dimensions expressed as a riser/tread count formula.",
            "properties": {
                "count": {"type": "number", "description": "Number of risers/treads, e.g. 15"},
                "riser_or_tread_dim": {"type": "number", "description": "Height/depth per riser or tread, e.g. 175"},
                "calculated_total": {"type": "number", "description": "count * riser_or_tread_dim, e.g. 2625"},
            },
        },
        "confidence": {
            "type": "string",
            "enum": CONFIDENCE_LEVELS,
            "description": "'low' if this reading is uncertain or illegible; 'high' if clearly legible.",
        },
        "notes": {
            "type": "string",
            "description": (
                "Context for an uncertain/illegible reading, or a math cross-check discrepancy "
                "(e.g. step_formula total doesn't match label_text, or floor-to-floor rise doesn't "
                "match the difference between the corresponding FFL datums)."
            ),
        },
    },
    "required": ["label_text"],
}

_ELEVATION_DATUM_ITEM_SCHEMA = {
    "type": "object",
    "properties": {
        "section_part": {
            "type": "string",
            "description": "Where this datum is called out, e.g. 'Top Exit Landing Threshold'.",
        },
        "level_code": {"type": "string", "description": "e.g. FFL, TOS, TOC, SSL"},
        "elevation_value": {"type": "number", "description": "e.g. 605.452"},
        "unit": {"type": "string", "description": "e.g. m"},
        "label_text": {
            "type": "string",
            "description": "Raw text as printed, e.g. '605.452 F.F.L.'. Use 'unclear' if illegible.",
        },
        "confidence": {
            "type": "string",
            "enum": CONFIDENCE_LEVELS,
            "description": "'low' if this reading is uncertain or illegible; 'high' if clearly legible.",
        },
        "notes": {
            "type": "string",
            "description": "Context for an uncertain/illegible reading.",
        },
    },
    "required": ["label_text"],
}

_DETAIL_CALLOUT_ITEM_SCHEMA = {
    "type": "object",
    "properties": {
        "callout_number": {"type": "string", "description": "The number/tag inside the circle/bubble, e.g. '6'."},
        "title": {"type": "string", "description": "The callout's label text, e.g. 'STEEL HANDRAIL DETAIL-1-5'."},
        "target_drawing_number": {
            "type": "string",
            "description": "Referenced sheet/drawing number printed under the callout, if shown, e.g. 'XXX-DWG-AR-AR-510002'.",
        },
        "section_part": {
            "type": "string",
            "description": "Where on the drawing this callout is, e.g. 'Top of upper flight, near grid 4'.",
        },
        "label_text": {
            "type": "string",
            "description": "Raw text as printed, e.g. '6 STEEL HANDRAIL DETAIL-1-5'. Use 'unclear' if illegible.",
        },
        "confidence": {
            "type": "string",
            "enum": CONFIDENCE_LEVELS,
            "description": "'low' if this reading is uncertain or illegible; 'high' if clearly legible.",
        },
        "notes": {
            "type": "string",
            "description": "Context for an uncertain/illegible reading.",
        },
    },
    "required": ["label_text"],
}

_QUANTITY_FIELD_SCHEMA = {
    "type": "object",
    "properties": {
        "value": {"type": "number", "description": "Numeric value in `unit`. Omit if not derivable from this drawing."},
        "unit": {"type": "string", "description": "e.g. mm, m, m2, m3"},
        "method": {
            "type": "string",
            "description": (
                "How this was obtained, e.g. 'count of stair_rise dimensions', 'sum of riser "
                "counts across flights', 'FFL(top) - FFL(bottom)', or 'not derivable -- requires "
                "plan view showing wall centerlines/thickness'."
            ),
        },
        "confidence": {
            "type": "string",
            "enum": CONFIDENCE_LEVELS,
            "description": "'low' when derived/estimated rather than directly labeled on the drawing.",
        },
        "notes": {
            "type": "string",
            "description": "Caveats, ambiguity, or what additional drawing (e.g. plan view) would be needed.",
        },
    },
}

_QUANTITY_TAKEOFF_FIELDS = [
    "num_doors", "num_drains",
    "num_flights", "num_risers_total", "riser_height", "num_treads_total", "tread_length",
    "total_tread_length", "num_landings", "total_vertical_drop", "perimeter_wall_length",
    "internal_room_footprint_area", "inner_perimeter", "wall_thickness", "centerline_perimeter",
    "perimeter_wall_concrete_quantity", "flight_landing_concrete_quantity", "wall_formwork_area",
    "soffit_stair_formwork_area",
]

_QUANTITY_TAKEOFF_SCHEMA = {
    "type": "object",
    "description": (
        "Stair/enclosure quantity-takeoff figures derived from this drawing's own dimensions "
        "and elevation_datums (e.g. flights/risers/treads/landings/vertical drop from a section), "
        "plus wall/footprint/concrete/formwork figures where this drawing is a plan view or "
        "detail that actually shows wall centerlines, thickness, or member sizes. Never invent a "
        "figure a plan view would be needed for -- set method to explain what's missing instead."
    ),
    "properties": {field: _QUANTITY_FIELD_SCHEMA for field in _QUANTITY_TAKEOFF_FIELDS},
}

_BOUNDING_BOX_SCHEMA = {
    "type": "object",
    "description": (
        "This drawing's extent on the full page, as fractions (0.0-1.0) of the page's total "
        "width/height, with (0,0) at the top-left corner of the page and (1,1) at the "
        "bottom-right. Used later to crop and re-render this drawing alone at higher "
        "resolution, so bound it generously (a few percent of padding) rather than tightly -- "
        "cutting off a dimension string at the edge is worse than including extra margin."
    ),
    "properties": {
        "x0": {"type": "number", "minimum": 0.0, "maximum": 1.0, "description": "Left edge, fraction of page width"},
        "y0": {"type": "number", "minimum": 0.0, "maximum": 1.0, "description": "Top edge, fraction of page height"},
        "x1": {"type": "number", "minimum": 0.0, "maximum": 1.0, "description": "Right edge, fraction of page width"},
        "y1": {"type": "number", "minimum": 0.0, "maximum": 1.0, "description": "Bottom edge, fraction of page height"},
        "confidence": {
            "type": "string",
            "enum": CONFIDENCE_LEVELS,
            "description": "'low' if this drawing's extent/border is ambiguous (e.g. no clear frame, overlaps a neighboring drawing).",
        },
        "notes": {"type": "string", "description": "Context, e.g. why the extent is uncertain or how much padding was added."},
    },
    "required": ["x0", "y0", "x1", "y1"],
}

TOOL_SCHEMA = {
    "name": TOOL_NAME,
    "description": (
        "Record every distinct, separately-framed drawing visible on this sheet/page "
        "(a single page can contain multiple drawings, e.g. a section next to an "
        "isometric view), each with its own metadata, every dimension annotation found "
        "within it, and every elevation datum/level callout (e.g. 'FFL 605.452') found within it."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "drawings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "drawing_type": {
                            "type": "string",
                            "enum": DRAWING_TYPES,
                            "description": "Best-fitting category for this drawing.",
                        },
                        "drawing_metadata": {
                            "type": "object",
                            "description": "Title-block / label info for this specific drawing, if visible.",
                            "properties": {
                                "project_name": {"type": "string"},
                                "drawing_title": {
                                    "type": "string",
                                    "description": (
                                        "This drawing's OWN title-callout text, e.g. 'Stair Details / "
                                        "Stair 00801' -- read from its title_callout_number circle, "
                                        "never from an internal equipment/room tag that happens to "
                                        "appear inside the drawing's own geometry."
                                    ),
                                },
                                "title_callout_number": {
                                    "type": "string",
                                    "description": (
                                        "The number inside THIS drawing's own title-callout circle "
                                        "(usually bottom-left of its frame), e.g. '4'. On a sheet of N "
                                        "drawings these are typically sequential 1..N, one per drawing."
                                    ),
                                },
                                "drawing_number": {
                                    "type": "string",
                                    "description": "Drawing/sheet number as printed, e.g. '2738-S13-H-26-S-512'.",
                                },
                                "sheet_scale": {"type": "string", "description": "e.g. '1:50'"},
                                "default_units": {
                                    "type": "string",
                                    "description": "Predominant unit used on this drawing, e.g. 'mm'.",
                                },
                                "revision": {
                                    "type": "string",
                                    "description": "Revision code/status from the title block, if visible.",
                                },
                            },
                        },
                        "dimensions": {
                            "type": "array",
                            "description": "Every dimension annotated on this drawing.",
                            "items": _DIMENSION_ITEM_SCHEMA,
                        },
                        "elevation_datums": {
                            "type": "array",
                            "description": "Every elevation/level datum called out on this drawing (e.g. FFL values).",
                            "items": _ELEVATION_DATUM_ITEM_SCHEMA,
                        },
                        "detail_callouts": {
                            "type": "array",
                            "description": (
                                "Every numbered detail-bubble cross-reference on this drawing (e.g. a "
                                "circled '6' pointing to 'STEEL HANDRAIL DETAIL-1-5' on another sheet). "
                                "These are real annotated content -- record them here rather than "
                                "dropping them just because they aren't a dimension or elevation datum."
                            ),
                            "items": _DETAIL_CALLOUT_ITEM_SCHEMA,
                        },
                        "quantity_takeoff": _QUANTITY_TAKEOFF_SCHEMA,
                        "bounding_box": _BOUNDING_BOX_SCHEMA,
                    },
                    "required": ["drawing_type", "dimensions", "bounding_box"],
                },
            }
        },
        "required": ["drawings"],
    },
}

SYSTEM_PROMPT = (
    "You are an expert Senior Architectural & Structural BIM Engineer and Technical Drawing "
    "Reader. Exhaustively extract and annotate EVERY visible dimension, datum level, and "
    "measurement string on this sheet. A single page can contain multiple physically distinct "
    "drawings (separated by border lines, whitespace, or separate title callouts) -- treat each "
    "one separately, with its own metadata/dimensions/elevation_datums. Do NOT create a separate "
    "'drawing' entry for the sheet's title block, revision table, key-plan/locator, north arrow, "
    "or general notes column -- that content belongs to the whole sheet, not to any one scaled "
    "drawing, and reporting it as its own drawing produces an empty or meaningless entry. Only "
    "segment actual scaled drawings (plan, section, elevation, detail, isometric, schedule).\n\n"
    "Multiple drawings on this sheet may look alike (e.g. several similar stair-plan or "
    "landing-plan views side by side) -- when reading each one's own title/label text, read it "
    "from THAT drawing's own title callout, never from a neighboring drawing's, even if the OCR "
    "text block handed to you contains both. If a drawing's title is genuinely illegible, set it "
    "to 'unclear' rather than guessing a neighbor's title, and cross-check that the title you did "
    "read is consistent with that drawing's own dimensions/labels (e.g. riser/tread numbers, "
    "landing name) before finalizing it.\n\n"
    "Follow this protocol for each drawing:\n"
    "1. Locate its bounding box FIRST, before reading any dimensions: find the drawing's outer "
    "border/frame (or, if it has no drawn border, the tightest rectangle enclosing all of its "
    "geometry, dimension lines, and title). Report x0/y0/x1/y1 as fractions of the FULL page's "
    "width/height (0,0 = top-left corner of the whole page, 1,1 = bottom-right), not fractions "
    "of some other region. Pad it generously (roughly 2-5% of page width/height on each side) so "
    "nothing near the edge is clipped -- this box will be used to crop and re-render this exact "
    "drawing alone at much higher resolution in a later pass, so a slightly loose box is fine but "
    "a tight/clipped one will cut off real content. If two drawings are packed tightly together "
    "with no gap (e.g. stacked in a column), split the boundary between them rather than "
    "overlapping their interiors -- each drawing's own title/number callout (usually printed at "
    "the bottom of its frame) belongs inside THAT drawing's own box, not the box of the drawing "
    "below it, so make sure your padding on a shared edge does not creep into the neighboring "
    "drawing's title text.\n"
    "2. Metadata scan -- find THIS drawing's own title callout: a distinctly bold/large number "
    "inside a circle or box (usually at the bottom-left of this drawing's own frame), immediately "
    "followed by this drawing's name (e.g. '(4) STAIR-07-INTERMEDIATE LANDING-03'). On a sheet "
    "with N drawings, these callout numbers are typically sequential across the whole sheet (1, 2, "
    "3, ... up to N), one per drawing -- record that number as `title_callout_number` and the text "
    "beside it as `drawing_title`. Do NOT confuse this with:\n"
    "   - an internal equipment/room identifier tag printed INSIDE the drawing's geometry (e.g. a "
    "small label like 'EGRESS STAIR 07 BG1 03 C' next to a door or room) -- that is a label for a "
    "component within the drawing, not the drawing's own title, even though both can look like "
    "short bold text;\n"
    "   - a `detail_callouts` bubble (a numbered circle pointing to a detail on another sheet) -- "
    "those are handled separately in step 3 below and are NOT this drawing's own title, even when "
    "their number happens to coincide with another drawing's title_callout_number;\n"
    "   - a grid-line bubble (e.g. a plain circled 'A' or '3' marking a column/row on the drawing) "
    "-- these have no title text beside them and are not a title callout.\n"
    "If you cannot find a bold numbered title callout for this drawing at all, set drawing_title to "
    "'unclear' and title_callout_number to null rather than reusing a neighboring drawing's number "
    "or an internal tag. Also record the drawing/sheet number, scale, default units (mm or m), and "
    "revision status from the title block.\n"
    "3. Spatial categorization -- systematically sweep the whole drawing:\n"
    "   - Exterior dimension strings (left, right, top, bottom)\n"
    "   - Interior compartment/shaft clear dimensions and headroom\n"
    "   - Vertical elevation datums (F.F.L., T.O.S., T.O.C., S.S.L.)\n"
    "   - Component details (stairs, handrails, doors, nosings, wall thicknesses, tread going)\n"
    "   - Detail-bubble cross-references: a circled/numbered tag (e.g. a circle containing '6') "
    "next to a title (e.g. 'STEEL HANDRAIL DETAIL-1-5') and often a small referenced drawing/sheet "
    "number underneath -- record every one of these as a `detail_callouts` entry (not as a "
    "dimension); they point to a detail shown elsewhere and are real sheet content worth keeping.\n"
    "4. Trace witness/extension lines: for every numerical figure, trace its bounding witness "
    "lines or datum markers to determine the precise start_reference and end_reference.\n"
    "5. Independent math cross-check: verify step-rise formulas (count * riser_or_tread_dim == "
    "calculated_total, and that this equals the annotated label_text); verify floor-to-floor "
    "rises against the difference between the corresponding FFL datums. If a check fails, still "
    "report the value as printed but note the discrepancy in `notes`.\n"
    "6. No-hallucination rule: never invent a dimension or value not actually shown. If text is "
    "illegible due to resolution or blur, set label_text to 'unclear' (or your best reading), "
    "set confidence to 'low', and explain the surrounding context in `notes`. Use the provided "
    "OCR text to help disambiguate anything hard to read in the image.\n"
    "7. Quantity takeoff: after recording dimensions/datums, derive `quantity_takeoff` for this "
    "drawing:\n"
    "   - num_doors: count every distinct door symbol/leaf visible on this drawing (typically a "
    "plan or detail view) -- a door swing arc, a door leaf line across an opening, or a door tag "
    "(e.g. 'D1', 'HD-01'). Count each physical door once even if it also has a dimension or tag "
    "labeling it. If this drawing is a section/elevation with no doors shown, leave it null with "
    "method 'not derivable -- no doors visible on this drawing'.\n"
    "   - num_drains: count every distinct drain symbol/tag visible on this drawing (e.g. a floor "
    "drain circle, gully, or a tag like 'FD-01'). Leave it null with an explanatory method if none "
    "are shown on this drawing.\n"
    "   - From a section/elevation showing stair_rise dimensions and FFL datums, derive: "
    "num_flights (count of stair_rise entries), num_risers_total (sum of their counts), "
    "riser_height (typical/per-flight riser dim), num_treads_total (risers - 1 per flight, "
    "summed), num_landings (distinct intermediate FFL levels between flights), and "
    "total_vertical_drop = the HIGHEST elevation_datum value you recorded for this drawing minus "
    "the LOWEST one -- use the actual max/min of this drawing's own `elevation_datums` list, never "
    "an arbitrary or intermediate pair of FFLs, even if one of them is labeled as a landing near "
    "the top; cross-check the result against the sum of all riser rises (they should match). "
    "tread_length/total_tread_length need a horizontal run or plan dimension; only fill "
    "them if one is actually shown (e.g. a diagonal flight-run dimension combined with the "
    "known rise), and mark the method/derivation used.\n"
    "   - perimeter_wall_length, internal_room_footprint_area, inner_perimeter, wall_thickness, "
    "centerline_perimeter, and the concrete/formwork quantities need a plan view (footprint, "
    "wall centerlines and thickness) plus, for volumes/formwork, member thickness or height. "
    "Only fill these from a plan/detail drawing that actually shows that geometry; on a "
    "section-only drawing leave them null and set `method` to state what's missing (e.g. "
    "'requires plan view showing wall centerlines/thickness -- not present in this section').\n"
    "   - Every quantity_takeoff field carries confidence/notes like a dimension does: 'low' for "
    "anything derived/estimated rather than read directly, with the arithmetic or assumption "
    "spelled out in notes.\n\n"
    "Call the record_drawings tool exactly once with your complete findings for this page."
)


def _build_client(settings: Settings) -> Anthropic:
    # Azure AI Foundry's Anthropic-compatible surface lives under an
    # `/anthropic` path on the resource endpoint (not the resource root), and
    # it authenticates with a plain `api-key` header rather than the
    # `x-api-key` header the Anthropic SDK sends by default for `api_key=`.
    # We keep `api_key=` too (harmless extra `X-Api-Key` header, and it also
    # keeps the SDK from trying its own credential auto-discovery), and add
    # the header Azure actually checks via `default_headers`.
    base_url = settings.foundry_endpoint.rstrip("/")
    if not base_url.endswith("/anthropic"):
        base_url = f"{base_url}/anthropic"
    return Anthropic(
        base_url=base_url,
        api_key=settings.foundry_api_key,
        default_headers={"api-key": settings.foundry_api_key},
    )


MAX_ATTEMPTS = 3


def _validate_drawings(raw: object) -> list[dict]:
    """Raise a descriptive ValueError if the tool call's 'drawings' field
    doesn't match the schema shape (e.g. a model returning plain strings
    instead of objects) -- some Foundry deployments don't strictly enforce
    tool input schemas the way the native Anthropic API does."""
    if not isinstance(raw, list):
        raise ValueError(f"'drawings' must be a JSON array, got {type(raw).__name__}: {raw!r}")

    for item in raw:
        if not isinstance(item, dict):
            raise ValueError(
                "each entry in 'drawings' must be a JSON object with "
                f"drawing_type/drawing_metadata/dimensions, got {type(item).__name__}: {item!r}"
            )

        dimensions = item.get("dimensions", [])
        if not isinstance(dimensions, list) or any(not isinstance(d, dict) for d in dimensions):
            raise ValueError(f"'dimensions' must be an array of JSON objects, got: {dimensions!r}")
        for dim in dimensions:
            if "label_text" not in dim:
                raise ValueError(f"each dimension must include 'label_text', got: {dim!r}")

        elevation_datums = item.get("elevation_datums", [])
        if not isinstance(elevation_datums, list) or any(not isinstance(d, dict) for d in elevation_datums):
            raise ValueError(f"'elevation_datums' must be an array of JSON objects, got: {elevation_datums!r}")

        detail_callouts = item.get("detail_callouts", [])
        if not isinstance(detail_callouts, list) or any(not isinstance(d, dict) for d in detail_callouts):
            raise ValueError(f"'detail_callouts' must be an array of JSON objects, got: {detail_callouts!r}")

        drawing_metadata = item.get("drawing_metadata", {})
        if drawing_metadata is not None and not isinstance(drawing_metadata, dict):
            raise ValueError(f"'drawing_metadata' must be a JSON object, got: {drawing_metadata!r}")

        quantity_takeoff = item.get("quantity_takeoff")
        if quantity_takeoff is not None and not isinstance(quantity_takeoff, dict):
            raise ValueError(f"'quantity_takeoff' must be a JSON object, got: {quantity_takeoff!r}")

        bounding_box = item.get("bounding_box")
        if bounding_box is not None:
            if not isinstance(bounding_box, dict):
                raise ValueError(f"'bounding_box' must be a JSON object, got: {bounding_box!r}")
            missing = [k for k in ("x0", "y0", "x1", "y1") if k not in bounding_box]
            if missing:
                raise ValueError(f"'bounding_box' is missing {missing}, got: {bounding_box!r}")

    return raw


def _build_messages(image_b64: str, ocr_text: str, page_number: int) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": image_b64,
                    },
                },
                {
                    "type": "text",
                    "text": (
                        f"Page {page_number} of the drawing sheet.\n\n"
                        "OCR text extracted from this page (may include noise or be "
                        "incomplete around graphics; use it to verify dimension readings):\n"
                        f"---\n{ocr_text or '(no OCR text extracted)'}\n---"
                    ),
                },
            ],
        }
    ]


def extract_page(image_bytes: bytes, ocr_text: str, page_number: int, settings: Settings) -> list[dict]:
    """Call Claude on one rendered page image; return raw drawing dicts
    (drawing_type/title/dimensions), unvalidated and without IDs assigned.

    Retries up to MAX_ATTEMPTS times. If the model's tool call doesn't match
    the expected shape, the error is fed back to it as a tool_result so it
    gets a chance to self-correct, rather than blindly repeating the same
    request. If every attempt fails, logs the problem and returns an empty
    list so one bad page doesn't take down the whole run.
    """
    client = _build_client(settings)
    image_b64 = base64.b64encode(image_bytes).decode("ascii")
    messages = _build_messages(image_b64, ocr_text, page_number)

    last_error: Exception | None = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            message = client.messages.create(
                model=settings.foundry_claude_deployment,
                max_tokens=8192,
                system=SYSTEM_PROMPT,
                tools=[TOOL_SCHEMA],
                tool_choice={"type": "tool", "name": TOOL_NAME},
                messages=messages,
            )
        except Exception as exc:
            logger.warning(
                "Page %s attempt %s: request failed: %s", page_number, attempt, exc
            )
            last_error = exc
            continue

        tool_block = next(
            (b for b in message.content if b.type == "tool_use" and b.name == TOOL_NAME),
            None,
        )
        if tool_block is None:
            last_error = ValueError("Claude did not return the expected record_drawings tool call")
            logger.warning("Page %s attempt %s: %s", page_number, attempt, last_error)
            continue

        try:
            return _validate_drawings(tool_block.input.get("drawings", []))
        except ValueError as exc:
            logger.warning("Page %s attempt %s: malformed tool call: %s", page_number, attempt, exc)
            last_error = exc
            # Feed the mistake back so the model can self-correct on the next attempt.
            messages.append({"role": "assistant", "content": message.content})
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_block.id,
                            "content": (
                                f"Your record_drawings call was invalid: {exc}. Call it again, "
                                "strictly matching the schema: 'drawings' is an array of objects "
                                "(never plain strings), each with drawing_type/drawing_metadata/"
                                "dimensions/elevation_datums/bounding_box, 'dimensions' is an array "
                                "of objects (never plain strings) each with at least 'label_text', "
                                "and 'bounding_box' is an object with numeric x0/y0/x1/y1."
                            ),
                            "is_error": True,
                        }
                    ],
                }
            )

    logger.error(
        "Giving up on page %s after %s attempts (%s); returning no drawings for this page",
        page_number, MAX_ATTEMPTS, last_error,
    )
    return []
