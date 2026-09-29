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
import copy
import json
import logging
import re

from anthropic import Anthropic

from .config import Settings
from .pricing import estimate_cost_usd
from .schema import LlmCallUsage

logger = logging.getLogger(__name__)

DRAWING_TYPES = ["elevation", "section", "isometric", "plan", "detail", "schedule", "other"]
ORIENTATIONS = ["vertical", "horizontal", "diagonal", "other"]
DIMENSION_TYPES = [
    "clearance", "headroom", "floor_to_floor", "level_difference", "guardrail_height",
    "stair_rise", "tread_going", "stair_width", "wall_thickness", "width", "other",
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

"You are an expert Technical Drawing Reader covering architectural, structural, civil, and "
    "MEP disciplines. Exhaustively extract and annotate EVERY visible dimension, datum level, and "
    "measurement string on this sheet, whatever discipline or project type it belongs to. A single "
    "page can contain multiple physically distinct drawings (separated by border lines, whitespace, "
    "or separate title callouts) -- treat each one separately, with its own metadata/dimensions/"
    "elevation_datums. Do NOT create a separate 'drawing' entry for the sheet's title block, "
    "revision table, key-plan/locator, north arrow, or general notes column -- that content belongs "
    "to the whole sheet, not to any one scaled drawing, and reporting it as its own drawing produces "
    "an empty or meaningless entry. Only segment actual scaled drawings (plan, section, elevation, "
    "detail, isometric, schedule).\n\n"
    "Multiple drawings on this sheet may look alike (e.g. several similar plan or landing views side "
    "by side) -- when reading each one's own title/label text, read it from THAT drawing's own title "
    "callout, never from a neighboring drawing's, even if the OCR text block handed to you contains "
    "both. If a drawing's title is genuinely illegible, set it to 'unclear' rather than guessing a "
    "neighbor's title, and cross-check that the title you did read is consistent with that drawing's "
    "own dimensions/labels before finalizing it.\n\n"
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
    "followed by this drawing's name (e.g. '(4) FOUNDATION PLAN - GRID A-D'). On a sheet with N "
    "drawings, these callout numbers are typically sequential across the whole sheet (1, 2, 3, ... "
    "up to N), one per drawing -- record that number as `title_callout_number` and the text beside "
    "it as `drawing_title`. Do NOT confuse this with:\n"
    "   - an internal equipment/room identifier tag printed INSIDE the drawing's geometry (e.g. a "
    "small label next to a door, duct, or room) -- that is a label for a component within the "
    "drawing, not the drawing's own title, even though both can look like short bold text;\n"
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
    "   - Component details relevant to this drawing's own discipline (stairs, handrails, doors, "
    "nosings, wall thicknesses, tread going, footings, rebar, ducts, pipes, pavement layers, etc.) "
    "-- tag each with the closest matching `type` from the controlled vocabulary, or `other` if "
    "none fit; never force a dimension into a type it doesn't actually match.\n"
    "   - Stair width (ONLY when this drawing shows a stair flight in plan): the CLEAR WIDTH of the "
    "flight itself -- the dimension spanning across the flight between its two bounding wall faces "
    "(or wall-to-handrail), measured perpendicular to the direction of travel. Tag this "
    "`stair_width`. Do NOT tag as `stair_width`: landing width/depth, overall room or enclosure "
    "width, corridor width, or wall thickness -- record those under `width`/`wall_thickness`/"
    "`other` instead.\n"
    "     Decide which of several nearby candidates is the real stair_width by what its witness/"
    "extension lines actually TOUCH, never by which number is bigger or smaller. Trace each "
    "candidate's start_reference/end_reference independently; a candidate is `stair_width` only if "
    "BOTH ends touch that one flight's own immediate bounding wall/handrail faces -- if either end "
    "touches a landing rail edge, a mid-point between flights, or spans the full enclosure/shaft "
    "across both flights, it is not stair_width, regardless of its magnitude relative to other "
    "nearby figures.\n"
    "     Don't silently discard a plausible candidate in favor of another for the same flight -- "
    "tag every reading whose witness lines plausibly bound a flight's clear width as `stair_width`, "
    "with `confidence` reflecting how sure you are. A drawing showing two stacked flights normally "
    "has TWO separate stair_width dimensions, one per flight, which can legitimately differ -- tag "
    "both individually and set `section_part` to say which flight each belongs to. Don't invent a "
    "theory that one flight's width is split by a central newel/handrail unless you can actually "
    "see that symbol drawn between the two witness lines.\n"
    "   - Detail-bubble cross-references: a circled/numbered tag (e.g. a circle containing '6') "
    "next to a title (e.g. 'STEEL HANDRAIL DETAIL-1-5') and often a small referenced drawing/sheet "
    "number underneath -- record every one of these as a `detail_callouts` entry (not as a "
    "dimension); they point to a detail shown elsewhere and are real sheet content worth keeping. "
    "Do NOT confuse this with a small triangular arrow marker (often at the drawing's left edge, "
    "pointing off the page) that references THIS SHEET's own drawing number -- that triangle marker "
    "is a sheet-navigation aid, not a detail callout, and its number/target must never be copied "
    "onto a nearby circular detail-callout bubble or vice versa; read each one's own number/target "
    "independently even when they sit close together.\n"
    "4. Quantity-takeoff-friendly labeling: a later pass matches dimensions to takeoff formulas by "
    "literally scanning each dimension's own `element`/`section_part`/`label_text` for shared words "
    "-- so how you word those fields determines whether this dimension can ever be used in a "
    "quantity takeoff, not just whether a human can read it. For every dimension:\n"
    "   - Write `element` as '<component> <measure>' in plain, concrete words a specification would "
    "use, e.g. 'Footing width', 'Wall height', 'Duct length', 'Slab thickness', 'Handrail length' -- "
    "not a code or abbreviation ('W1'), and not just the bare measure ('Width') without saying what "
    "it belongs to.\n"
    "   - Also record every countable repeated element as its own `dimensions` entry with `type` "
    "'other', `value` set to the count, `unit` 'count' or 'no.', and `element` naming what's being "
    "counted, e.g. element='Footing count', value=6, unit='count' -- for footings, columns, doors, "
    "windows, fixtures, or any other element repeated on this drawing where a total count is "
    "determinable (from a schedule table, repeated symbols, or an explicit callout like '6 No. "
    "FOOTINGS'). Do not skip these just because they aren't a dimension LINE -- a count is as real a "
    "quantity-takeoff input as a length.\n"
    "   - For a linear run relevant to a takeoff (a wall's length along its own run in plan, a "
    "duct/pipe run, a kerb/handrail run), record it explicitly even if it seems redundant with the "
    "drawing's overall extent -- e.g. element='Wall length (Grid A-D)', not left implicit in the "
    "drawing's bounding box.\n\n"
    "5. Trace witness/extension lines: for every numerical figure, trace its bounding witness "
    "lines or datum markers to determine the precise start_reference and end_reference.\n"
    "6. Independent math cross-check: verify step-rise formulas (count * riser_or_tread_dim == "
    "calculated_total, and that this equals the annotated label_text); verify floor-to-floor "
    "rises against the difference between the corresponding FFL datums. If a check fails, still "
    "report the value as printed but note the discrepancy in `notes`.\n"
    "7. No-hallucination rule: never invent a dimension or value not actually shown. If text is "
    "illegible due to resolution or blur, set label_text to 'unclear' (or your best reading), "
    "set confidence to 'low', and explain the surrounding context in `notes`. Use the provided "
    "OCR text to help disambiguate anything hard to read in the image.\n"
    "8. If this image is a tightly cropped, high-resolution re-render of a single drawing (rather "
    "than a full sheet), treat that as your best chance to resolve any ambiguous reading precisely "
    "-- use the extra resolution to trace witness lines pixel-by-pixel rather than repeating a "
    "lower-confidence guess from a wider view.\n\n"
    "Call the record_drawings tool exactly once with your complete findings for this page."

    #old 2


    # "You are an expert Technical Drawing Reader covering architectural, structural, civil, and "
    # "MEP disciplines. Exhaustively extract and annotate EVERY visible dimension, datum level, and "
    # "measurement string on this sheet, whatever discipline or project type it belongs to. A single "
    # "page can contain multiple physically distinct drawings (separated by border lines, whitespace, "
    # "or separate title callouts) -- treat each one separately, with its own metadata/dimensions/"
    # "elevation_datums. Do NOT create a separate 'drawing' entry for the sheet's title block, "
    # "revision table, key-plan/locator, north arrow, or general notes column -- that content belongs "
    # "to the whole sheet, not to any one scaled drawing, and reporting it as its own drawing produces "
    # "an empty or meaningless entry. Only segment actual scaled drawings (plan, section, elevation, "
    # "detail, isometric, schedule).\n\n"
    # "Multiple drawings on this sheet may look alike (e.g. several similar plan or landing views side "
    # "by side) -- when reading each one's own title/label text, read it from THAT drawing's own title "
    # "callout, never from a neighboring drawing's, even if the OCR text block handed to you contains "
    # "both. If a drawing's title is genuinely illegible, set it to 'unclear' rather than guessing a "
    # "neighbor's title, and cross-check that the title you did read is consistent with that drawing's "
    # "own dimensions/labels before finalizing it.\n\n"
    # "Follow this protocol for each drawing:\n"
    # "1. Locate its bounding box FIRST, before reading any dimensions: find the drawing's outer "
    # "border/frame (or, if it has no drawn border, the tightest rectangle enclosing all of its "
    # "geometry, dimension lines, and title). Report x0/y0/x1/y1 as fractions of the FULL page's "
    # "width/height (0,0 = top-left corner of the whole page, 1,1 = bottom-right), not fractions "
    # "of some other region. Pad it generously (roughly 2-5% of page width/height on each side) so "
    # "nothing near the edge is clipped -- this box will be used to crop and re-render this exact "
    # "drawing alone at much higher resolution in a later pass, so a slightly loose box is fine but "
    # "a tight/clipped one will cut off real content. If two drawings are packed tightly together "
    # "with no gap (e.g. stacked in a column), split the boundary between them rather than "
    # "overlapping their interiors -- each drawing's own title/number callout (usually printed at "
    # "the bottom of its frame) belongs inside THAT drawing's own box, not the box of the drawing "
    # "below it, so make sure your padding on a shared edge does not creep into the neighboring "
    # "drawing's title text.\n"
    # "2. Metadata scan -- find THIS drawing's own title callout: a distinctly bold/large number "
    # "inside a circle or box (usually at the bottom-left of this drawing's own frame), immediately "
    # "followed by this drawing's name (e.g. '(4) FOUNDATION PLAN - GRID A-D'). On a sheet with N "
    # "drawings, these callout numbers are typically sequential across the whole sheet (1, 2, 3, ... "
    # "up to N), one per drawing -- record that number as `title_callout_number` and the text beside "
    # "it as `drawing_title`. Do NOT confuse this with:\n"
    # "   - an internal equipment/room identifier tag printed INSIDE the drawing's geometry (e.g. a "
    # "small label next to a door, duct, or room) -- that is a label for a component within the "
    # "drawing, not the drawing's own title, even though both can look like short bold text;\n"
    # "   - a `detail_callouts` bubble (a numbered circle pointing to a detail on another sheet) -- "
    # "those are handled separately in step 3 below and are NOT this drawing's own title, even when "
    # "their number happens to coincide with another drawing's title_callout_number;\n"
    # "   - a grid-line bubble (e.g. a plain circled 'A' or '3' marking a column/row on the drawing) "
    # "-- these have no title text beside them and are not a title callout.\n"
    # "If you cannot find a bold numbered title callout for this drawing at all, set drawing_title to "
    # "'unclear' and title_callout_number to null rather than reusing a neighboring drawing's number "
    # "or an internal tag. Also record the drawing/sheet number, scale, default units (mm or m), and "
    # "revision status from the title block.\n"
    # "3. Spatial categorization -- systematically sweep the whole drawing:\n"
    # "   - Exterior dimension strings (left, right, top, bottom)\n"
    # "   - Interior compartment/shaft clear dimensions and headroom\n"
    # "   - Vertical elevation datums (F.F.L., T.O.S., T.O.C., S.S.L.)\n"
    # "   - Component details relevant to this drawing's own discipline (stairs, handrails, doors, "
    # "nosings, wall thicknesses, tread going, footings, rebar, ducts, pipes, pavement layers, etc.) "
    # "-- tag each with the closest matching `type` from the controlled vocabulary, or `other` if "
    # "none fit; never force a dimension into a type it doesn't actually match.\n"
    # "   - Stair width (ONLY when this drawing shows a stair flight in plan): the CLEAR WIDTH of the "
    # "flight itself -- the dimension spanning across the flight between its two bounding wall faces "
    # "(or wall-to-handrail), measured perpendicular to the direction of travel. Tag this "
    # "`stair_width`. Do NOT tag as `stair_width`: landing width/depth, overall room or enclosure "
    # "width, corridor width, or wall thickness -- record those under `width`/`wall_thickness`/"
    # "`other` instead.\n"
    # "     Decide which of several nearby candidates is the real stair_width by what its witness/"
    # "extension lines actually TOUCH, never by which number is bigger or smaller. Trace each "
    # "candidate's start_reference/end_reference independently; a candidate is `stair_width` only if "
    # "BOTH ends touch that one flight's own immediate bounding wall/handrail faces -- if either end "
    # "touches a landing rail edge, a mid-point between flights, or spans the full enclosure/shaft "
    # "across both flights, it is not stair_width, regardless of its magnitude relative to other "
    # "nearby figures.\n"
    # "     Don't silently discard a plausible candidate in favor of another for the same flight -- "
    # "tag every reading whose witness lines plausibly bound a flight's clear width as `stair_width`, "
    # "with `confidence` reflecting how sure you are. A drawing showing two stacked flights normally "
    # "has TWO separate stair_width dimensions, one per flight, which can legitimately differ -- tag "
    # "both individually and set `section_part` to say which flight each belongs to. Don't invent a "
    # "theory that one flight's width is split by a central newel/handrail unless you can actually "
    # "see that symbol drawn between the two witness lines.\n"
    # "   - Detail-bubble cross-references: a circled/numbered tag (e.g. a circle containing '6') "
    # "next to a title (e.g. 'STEEL HANDRAIL DETAIL-1-5') and often a small referenced drawing/sheet "
    # "number underneath -- record every one of these as a `detail_callouts` entry (not as a "
    # "dimension); they point to a detail shown elsewhere and are real sheet content worth keeping. "
    # "Do NOT confuse this with a small triangular arrow marker (often at the drawing's left edge, "
    # "pointing off the page) that references THIS SHEET's own drawing number -- that triangle marker "
    # "is a sheet-navigation aid, not a detail callout, and its number/target must never be copied "
    # "onto a nearby circular detail-callout bubble or vice versa; read each one's own number/target "
    # "independently even when they sit close together.\n"
    # "4. Trace witness/extension lines: for every numerical figure, trace its bounding witness "
    # "lines or datum markers to determine the precise start_reference and end_reference.\n"
    # "5. Independent math cross-check: verify step-rise formulas (count * riser_or_tread_dim == "
    # "calculated_total, and that this equals the annotated label_text); verify floor-to-floor "
    # "rises against the difference between the corresponding FFL datums. If a check fails, still "
    # "report the value as printed but note the discrepancy in `notes`.\n"
    # "6. No-hallucination rule: never invent a dimension or value not actually shown. If text is "
    # "illegible due to resolution or blur, set label_text to 'unclear' (or your best reading), "
    # "set confidence to 'low', and explain the surrounding context in `notes`. Use the provided "
    # "OCR text to help disambiguate anything hard to read in the image.\n"
    # "7. If this image is a tightly cropped, high-resolution re-render of a single drawing (rather "
    # "than a full sheet), treat that as your best chance to resolve any ambiguous reading precisely "
    # "-- use the extra resolution to trace witness lines pixel-by-pixel rather than repeating a "
    # "lower-confidence guess from a wider view.\n\n"
    # "Call the record_drawings tool exactly once with your complete findings for this page."


    #old 1
    # "You are an expert Senior Architectural & Structural BIM Engineer and Technical Drawing "
    # "Reader. Exhaustively extract and annotate EVERY visible dimension, datum level, and "
    # "measurement string on this sheet. A single page can contain multiple physically distinct "
    # "drawings (separated by border lines, whitespace, or separate title callouts) -- treat each "
    # "one separately, with its own metadata/dimensions/elevation_datums. Do NOT create a separate "
    # "'drawing' entry for the sheet's title block, revision table, key-plan/locator, north arrow, "
    # "or general notes column -- that content belongs to the whole sheet, not to any one scaled "
    # "drawing, and reporting it as its own drawing produces an empty or meaningless entry. Only "
    # "segment actual scaled drawings (plan, section, elevation, detail, isometric, schedule).\n\n"
    # "Multiple drawings on this sheet may look alike (e.g. several similar stair-plan or "
    # "landing-plan views side by side) -- when reading each one's own title/label text, read it "
    # "from THAT drawing's own title callout, never from a neighboring drawing's, even if the OCR "
    # "text block handed to you contains both. If a drawing's title is genuinely illegible, set it "
    # "to 'unclear' rather than guessing a neighbor's title, and cross-check that the title you did "
    # "read is consistent with that drawing's own dimensions/labels (e.g. riser/tread numbers, "
    # "landing name) before finalizing it.\n\n"
    # "Follow this protocol for each drawing:\n"
    # "1. Locate its bounding box FIRST, before reading any dimensions: find the drawing's outer "
    # "border/frame (or, if it has no drawn border, the tightest rectangle enclosing all of its "
    # "geometry, dimension lines, and title). Report x0/y0/x1/y1 as fractions of the FULL page's "
    # "width/height (0,0 = top-left corner of the whole page, 1,1 = bottom-right), not fractions "
    # "of some other region. Pad it generously (roughly 2-5% of page width/height on each side) so "
    # "nothing near the edge is clipped -- this box will be used to crop and re-render this exact "
    # "drawing alone at much higher resolution in a later pass, so a slightly loose box is fine but "
    # "a tight/clipped one will cut off real content. If two drawings are packed tightly together "
    # "with no gap (e.g. stacked in a column), split the boundary between them rather than "
    # "overlapping their interiors -- each drawing's own title/number callout (usually printed at "
    # "the bottom of its frame) belongs inside THAT drawing's own box, not the box of the drawing "
    # "below it, so make sure your padding on a shared edge does not creep into the neighboring "
    # "drawing's title text.\n"
    # "2. Metadata scan -- find THIS drawing's own title callout: a distinctly bold/large number "
    # "inside a circle or box (usually at the bottom-left of this drawing's own frame), immediately "
    # "followed by this drawing's name (e.g. '(4) STAIR-07-INTERMEDIATE LANDING-03'). On a sheet "
    # "with N drawings, these callout numbers are typically sequential across the whole sheet (1, 2, "
    # "3, ... up to N), one per drawing -- record that number as `title_callout_number` and the text "
    # "beside it as `drawing_title`. Do NOT confuse this with:\n"
    # "   - an internal equipment/room identifier tag printed INSIDE the drawing's geometry (e.g. a "
    # "small label like 'EGRESS STAIR 07 BG1 03 C' next to a door or room) -- that is a label for a "
    # "component within the drawing, not the drawing's own title, even though both can look like "
    # "short bold text;\n"
    # "   - a `detail_callouts` bubble (a numbered circle pointing to a detail on another sheet) -- "
    # "those are handled separately in step 3 below and are NOT this drawing's own title, even when "
    # "their number happens to coincide with another drawing's title_callout_number;\n"
    # "   - a grid-line bubble (e.g. a plain circled 'A' or '3' marking a column/row on the drawing) "
    # "-- these have no title text beside them and are not a title callout.\n"
    # "If you cannot find a bold numbered title callout for this drawing at all, set drawing_title to "
    # "'unclear' and title_callout_number to null rather than reusing a neighboring drawing's number "
    # "or an internal tag. Also record the drawing/sheet number, scale, default units (mm or m), and "
    # "revision status from the title block.\n"
    # "3. Spatial categorization -- systematically sweep the whole drawing:\n"
    # "   - Exterior dimension strings (left, right, top, bottom)\n"
    # "   - Interior compartment/shaft clear dimensions and headroom\n"
    # "   - Vertical elevation datums (F.F.L., T.O.S., T.O.C., S.S.L.)\n"
    # "   - Component details (stairs, handrails, doors, nosings, wall thicknesses, tread going)\n"
    # "   - Stair width: on a PLAN view, the CLEAR WIDTH of the stair flight itself -- the "
    # "dimension spanning across the flight between its two bounding wall faces (or wall-to-"
    # "handrail), measured perpendicular to the direction of travel. Tag this `stair_width`. "
    # "Do NOT tag as `stair_width`: landing width/depth, overall room or enclosure width, "
    # "corridor width, or wall thickness -- those are real dimensions worth recording too, "
    # "just under `width`/`wall_thickness`/`other` instead, not `stair_width`.\n"
    # "     Decide which of several nearby candidates is the real stair_width by what its "
    # "witness/extension lines actually TOUCH, never by which number is bigger or smaller -- "
    # "magnitude alone is not a reliable signal, and sometimes the larger of two adjacent "
    # "figures is the correct flight width while the smaller is a partial sub-segment, and "
    # "sometimes it's the reverse. For each candidate, trace both ends independently and set "
    # "start_reference/end_reference to what they concretely touch (e.g. 'inner wall face of "
    # "upper flight' / 'handrail centerline of upper flight'). A candidate is `stair_width` "
    # "only if BOTH ends touch that one flight's own immediate bounding wall/handrail faces; if "
    # "either end touches a landing rail edge, a mid-point between flights, an adjacent room, or "
    # "spans the full enclosure/shaft between the outer walls (i.e. across both flights or the "
    # "whole stairwell), it is not stair_width for that flight -- tag it `width` or `other` "
    # "instead, regardless of its magnitude relative to other nearby figures.\n"
    # "     Do not silently discard a plausible candidate just because you end up preferring "
    # "another one for the same flight -- tag EVERY reading whose witness lines plausibly "
    # "bound a flight's clear width as `stair_width` (not demoted to `width`/`other`), and set "
    # "its `confidence` to reflect how sure you are: 'high' only when both ends unambiguously "
    # "touch that flight's own wall/handrail faces, 'medium'/'low' when there's real doubt. This "
    # "keeps every candidate visible instead of losing one to a wrong either/or guess.\n"
    # "     A plan view showing two stacked flights (e.g. an upper flight and a lower flight one "
    # "above the other, as on an intermediate landing plan) normally has TWO separate stair_width "
    # "dimensions, one per flight, which can legitimately differ from each other -- look for and "
    # "tag both individually rather than assuming one value applies to the whole drawing, and set "
    # "that dimension's `section_part` to say which flight it belongs to (e.g. 'Upper flight' or "
    # "'Lower flight') so the two can be told apart later. A plan view showing only a single "
    # "flight only has one.\n"
    # "     Do not invent a theory that one flight's width is split into two segments by a "
    # "central newel/handrail unless you can actually see that newel/handrail symbol drawn "
    # "between the two witness lines in the image -- if you can't point to the physical symbol "
    # "causing the split, it is far more likely that a single continuous dimension line was "
    # "misread as two, so re-trace it as one reading spanning the whole flight instead.\n"
    # "     If this image is a tightly cropped, high-resolution re-render of a single drawing "
    # "(rather than a full sheet), treat that as your best chance to resolve any stair_width "
    # "ambiguity precisely -- use the extra resolution to trace witness lines pixel-by-pixel "
    # "rather than repeating a lower-confidence guess from a wider view.\n"
    # "   - Detail-bubble cross-references: a circled/numbered tag (e.g. a circle containing '6') "
    # "next to a title (e.g. 'STEEL HANDRAIL DETAIL-1-5') and often a small referenced drawing/sheet "
    # "number underneath -- record every one of these as a `detail_callouts` entry (not as a "
    # "dimension); they point to a detail shown elsewhere and are real sheet content worth keeping. "
    # "Do NOT confuse this with a small triangular arrow marker (often at the drawing's left edge, "
    # "pointing off the page) that references THIS SHEET's own drawing number -- that triangle marker "
    # "is a sheet-navigation aid, not a detail callout, and its number/target must never be copied "
    # "onto a nearby circular detail-callout bubble or vice versa; read each one's own number/target "
    # "independently even when they sit close together.\n"
    # "4. Trace witness/extension lines: for every numerical figure, trace its bounding witness "
    # "lines or datum markers to determine the precise start_reference and end_reference.\n"
    # "5. Independent math cross-check: verify step-rise formulas (count * riser_or_tread_dim == "
    # "calculated_total, and that this equals the annotated label_text); verify floor-to-floor "
    # "rises against the difference between the corresponding FFL datums. If a check fails, still "
    # "report the value as printed but note the discrepancy in `notes`.\n"
    # "6. No-hallucination rule: never invent a dimension or value not actually shown. If text is "
    # "illegible due to resolution or blur, set label_text to 'unclear' (or your best reading), "
    # "set confidence to 'low', and explain the surrounding context in `notes`. Use the provided "
    # "OCR text to help disambiguate anything hard to read in the image.\n\n"
    # "Call the record_drawings tool exactly once with your complete findings for this page."
    
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


def _record_usage(
    usage_sink: list[LlmCallUsage] | None,
    call_type: str,
    page_number: int,
    drawing_id: str | None,
    attempt: int,
    message: object,
    settings: Settings,
) -> None:
    """Append one LlmCallUsage record for a call that got a response,
    successful or not -- a malformed/retried reply still consumed tokens
    and costs money, so it must still be counted. No-op if the caller isn't
    tracking usage (usage_sink is None) or the response has no usage field."""
    if usage_sink is None:
        return
    usage = getattr(message, "usage", None)
    if usage is None:
        return
    input_tokens = getattr(usage, "input_tokens", 0) or 0
    output_tokens = getattr(usage, "output_tokens", 0) or 0
    usage_sink.append(
        LlmCallUsage(
            call_type=call_type,
            page_number=page_number,
            drawing_id=drawing_id,
            attempt=attempt,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=estimate_cost_usd(settings.foundry_claude_deployment, input_tokens, output_tokens),
        )
    )


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


_DYNAMIC_PROPERTIES_PROMPT_ADDENDUM = (
    "\n\n8. Resource-planning input roles: this drawing's own tool schema includes an extra "
    "`quantities` object under each drawing, whose properties are specific named input roles "
    "(e.g. 'wall_height', 'main_bar_diameter') proposed for this drawing's classification. For "
    "each role you can actually determine from what's shown, report its numeric value under that "
    "exact key -- read it the same careful way you read any other dimension (trace witness lines, "
    "check against OCR text), never guess. Omit a role entirely (or set it null) if this specific "
    "drawing doesn't show it; do not force a value onto a role that isn't really there."
)


def _tool_schema_with_dynamic_properties(dynamic_properties: dict[str, dict]) -> dict:
    """A copy of TOOL_SCHEMA with an extra `quantities` object property added
    to each drawing item -- one property per proposed resource-planning
    input role (see classify_drawing), so the SAME extraction pass that
    reads dimensions/datums/callouts can also report each role's value
    directly, instead of a separate downstream step guessing which
    dimension matches a role from free text after the fact. Python still
    does 100% of the arithmetic on these values (pipeline.calculations) --
    this only changes WHERE a raw input value gets read from, never who
    computes a formula over it.
    """
    schema = copy.deepcopy(TOOL_SCHEMA)
    item_properties = schema["input_schema"]["properties"]["drawings"]["items"]["properties"]
    item_properties["quantities"] = {
        "type": "object",
        "description": (
            "Numeric values for this drawing's own resource-planning input roles (proposed by "
            "the earlier classification pass for this drawing). One key per role below; omit or "
            "null any role this drawing doesn't actually show -- never fabricate a value."
        ),
        "properties": dynamic_properties,
    }
    return schema


def extract_page(
    image_bytes: bytes,
    ocr_text: str,
    page_number: int,
    settings: Settings,
    dynamic_properties: dict[str, dict] | None = None,
    usage_sink: list[LlmCallUsage] | None = None,
    call_type: str = "segmentation",
    drawing_id: str | None = None,
) -> list[dict]:
    """Call Claude on one rendered page image; return raw drawing dicts
    (drawing_type/title/dimensions), unvalidated and without IDs assigned.

    `dynamic_properties`, when given, merges an extra `quantities` object
    into the per-drawing tool schema for this call only (see
    _tool_schema_with_dynamic_properties) -- used for the per-drawing detail
    pass, once classify_drawing has already proposed this drawing's own
    resource-planning input roles. The base (no-argument) call used for
    Pass 1 whole-page segmentation is unaffected.

    `usage_sink`/`call_type`/`drawing_id`, when given, record one
    LlmCallUsage per attempt onto `usage_sink` (see _record_usage) --
    `call_type` should be "segmentation" for the whole-page Pass 1 call
    (the default; `drawing_id` stays None since one such call covers every
    drawing on the page) or "detail" for the per-drawing Pass 2 call.

    Retries up to MAX_ATTEMPTS times. If the model's tool call doesn't match
    the expected shape, the error is fed back to it as a tool_result so it
    gets a chance to self-correct, rather than blindly repeating the same
    request. If every attempt fails, logs the problem and returns an empty
    list so one bad page doesn't take down the whole run.
    """
    client = _build_client(settings)
    image_b64 = base64.b64encode(image_bytes).decode("ascii")
    messages = _build_messages(image_b64, ocr_text, page_number)

    tool_schema = (
        _tool_schema_with_dynamic_properties(dynamic_properties) if dynamic_properties else TOOL_SCHEMA
    )
    system_prompt = SYSTEM_PROMPT + _DYNAMIC_PROPERTIES_PROMPT_ADDENDUM if dynamic_properties else SYSTEM_PROMPT

    last_error: Exception | None = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            message = client.messages.create(
                model=settings.foundry_claude_deployment,
                max_tokens=8192,
                system=system_prompt,
                tools=[tool_schema],
                tool_choice={"type": "tool", "name": TOOL_NAME},
                messages=messages,
            )
        except Exception as exc:
            logger.warning(
                "Page %s attempt %s: request failed: %s", page_number, attempt, exc
            )
            last_error = exc
            continue

        _record_usage(usage_sink, call_type, page_number, drawing_id, attempt, message, settings)

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


# --- Drawing classification + dynamic takeoff-schema proposal --------------
#
# A second, separate plain-JSON prompt call per drawing crop (Pass 1.5,
# between segmentation and the detail pass) -- deliberately NOT a forced
# tool call (see classify_drawing's docstring): the model identifies what
# the drawing actually is, then proposes which resource-planning quantities
# can be derived from it and how -- but never computes them.
# pipeline.calculations does the arithmetic; pipeline.schema_registry turns
# the proposal into a validated JSON Schema and reuses it across drawings
# that share a classification; the proposed input roles then flow into
# extract_page's own tool schema (_tool_schema_with_dynamic_properties) so
# the main extraction pass can report their values directly.

MEASUREMENT_BASES = ["count", "length", "area", "volume", "weight"]
RESOURCE_CATEGORIES = ["material", "equipment"]

_CLASSIFICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "project_type": {"type": "string", "description": "e.g. 'commercial building', 'road/highway', 'tunnel'"},
        "discipline": {"type": "string", "description": "e.g. 'architectural', 'structural', 'MEP', 'civil'"},
        "drawing_type": {"type": "string", "enum": DRAWING_TYPES},
        "building_elements": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Building elements/systems shown, e.g. ['foundation', 'footing'] or ['stair', 'handrail']",
        },
        "confidence": {"type": "string", "enum": CONFIDENCE_LEVELS},
        "notes": {"type": "string", "description": "Context for an uncertain classification"},
    },
    "required": ["drawing_type", "confidence"],
}

_QUANTITY_ITEM_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "description": "Quantity item name, e.g. 'wall_concrete_volume'"},
        "description": {"type": "string"},
        "unit": {"type": "string", "description": "e.g. m3, m2, m, count, kg"},
        "measurement_basis": {"type": "string", "enum": MEASUREMENT_BASES},
        "resource_category": {"type": "string", "enum": RESOURCE_CATEGORIES},
        "depends_on": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Logical input role names THIS formula needs, e.g. ['wall_length', 'wall_height', "
                "'wall_thickness'] -- snake_case, scoped to this one drawing's own schema, NOT this "
                "drawing's own dimension_ids."
            ),
        },
        "input_units": {
            "type": "object",
            "description": (
                "The unit (mm/cm/m) each depends_on role's value must be converted to before this "
                "formula is evaluated, e.g. {'wall_length': 'm', 'wall_thickness': 'm', "
                "'main_bar_diameter': 'mm'} -- state whatever unit YOUR formula's own convention "
                "actually needs for each role (e.g. a rebar weight formula conventionally uses a "
                "millimetre bar diameter together with a metre bar length in the same expression); "
                "the value will be converted from whatever unit it was actually found in on the "
                "drawing into the unit you state here, so the formula always sees consistent units "
                "regardless of how the drawing itself is dimensioned."
            ),
        },
        "formula": {
            "type": "string",
            "description": (
                "A plain arithmetic expression over the `depends_on` names ONLY -- e.g. "
                "'wall_length * wall_height * wall_thickness'. Arithmetic operators and "
                "parentheses only: no function calls, no computed numbers, no units in the "
                "expression itself. You propose the formula; it is evaluated in Python, never by you."
            ),
        },
    },
    "required": ["name", "unit", "measurement_basis", "resource_category", "depends_on", "formula"],
}

# Kept as a human-readable reference for what shape the classification JSON
# must take (see the formatting instructions appended to
# CLASSIFICATION_SYSTEM_PROMPT below) -- no longer wired into the API call as
# a forced tool_choice. classify_drawing() now asks for this shape as plain
# JSON text and parses/validates it itself (_validate_classification), so a
# Foundry deployment doesn't need tool-calling support for this step at all.
CLASSIFY_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "classification": _CLASSIFICATION_SCHEMA,
        "quantity_items": {
            "type": "array",
            "description": (
                "Every quantity-takeoff item genuinely derivable from this drawing. Empty if "
                "nothing meaningful can be derived (e.g. a legend or a drawing with no "
                "dimensioned geometry) -- do not invent an item just to fill this array."
            ),
            "items": _QUANTITY_ITEM_SCHEMA,
        },
    },
    "required": ["classification", "quantity_items"],
}

CLASSIFICATION_SYSTEM_PROMPT = (
"You are an expert quantity surveyor/estimator reviewing ONE architectural/structural/civil/MEP "
    "drawing (already cropped from its sheet). This drawing could be ANY discipline or element type "
    "-- do not assume stairs, concrete, or any specific building system; read what is actually shown "
    "and derive a takeoff schema that fits THIS drawing alone. Two jobs, in order:\n\n"
    "1. Classify it: project_type (the kind of facility/project this drawing belongs to), discipline "
    "(architectural/structural/MEP/civil/etc.), drawing_type, and building_elements (the specific "
    "elements/systems actually shown -- e.g. ['stair', 'handrail'], ['foundation', 'footing', "
    "'rebar'], ['duct', 'diffuser'], ['road', 'pavement', 'curb'], ['door', 'window'], ['pipe', "
    "'valve'], ['pavement', 'kerb', 'drainage']). Set confidence to 'low' if the drawing's subject is "
    "genuinely ambiguous (e.g. a fragment with no title or legible content) rather than guessing.\n\n"
    "2. Propose a quantity-takeoff schema for THIS ONE DRAWING SPECIFICALLY: think "
    "like an estimator building a bill of quantities for  materials, and equipment. This schema "
    "is generated fresh for this drawing alone -- it is never shared, reused, or extended from "
    "another drawing's schema, even one classified identically, because two drawings with the same "
    "discipline/drawing_type can still show entirely different physical subjects (e.g. a full wall "
    "section vs. a small local recess detail).\n\n"
    "BE THOROUGH, NOT MINIMAL -- propose every quantity item that a real estimator would pull off "
    "this specific drawing, across whichever of these apply to what's actually shown (skip any "
    "category with nothing genuinely derivable; never force one in):\n"
    "   - count: discrete units shown or countable on this drawing, e.g. door_count, window_count, "
    "column_count, footing_count, fixture_count, bar_count.\n"
    "   - length: runs and linear members, e.g. wall_length, duct_length, pipe_length, "
    "handrail_length, kerb_length, bar_length.\n"
    "   - area: surfaces, e.g. wall_area, formwork_area, floor_finish_area, pavement_area, "
    "waterproofing_area, insulation_area -- usually derived from two of this drawing's own "
    "length/height/width dimensions, not a directly labeled area figure.\n"
    "   - volume: 3D quantities, e.g. concrete_volume, excavation_volume, backfill_volume, "
    "asphalt_volume -- from length x width x depth/thickness, or footprint area x depth.\n"
    "   - weight: material mass, e.g. rebar_weight, steel_section_weight -- only when the inputs to "
    "compute it are actually derivable from this drawing (see rebar note below); never invent a "
    "unit-weight/density constant to get there (see formula rule below).\n"
    "Propose ONLY items genuinely derivable from what THIS specific drawing itself shows -- an empty "
    "quantity_items list is correct and expected when nothing meaningful can be derived (e.g. a key "
    "plan, a legend, a fragment with no legible geometry, or a note-only sheet). Do not pad the list "
    "with an item just to have more entries.\n\n"
    "CRITICAL -- respect what this drawing's own drawing_type can physically show:\n"
    "   - A section or elevation is a cut THROUGH a member -- it can show that member's height, "
    "depth, and thickness (the dimensions visible in the cut plane), and it can show bar size/"
    "spacing callouts, but it CANNOT show that member's own length along its run (how far a wall or "
    "slab extends in plan) -- that is only visible in a plan view. Do not propose a role like "
    "'wall_length' or 'slab_length' on a section/elevation drawing; if a quantity genuinely needs a "
    "run length, either leave it out of quantity_items entirely, or note in the item's description "
    "that it requires a companion plan view this drawing does not provide.\n"
    "   - A rebar weight/length item needs the bar's own physical cut length as an input. A bar's "
    "size and spacing ARE visible on a section (its bar-mark callout), but its cut length almost "
    "never is -- do not propose a role for it unless this specific drawing actually shows a "
    "dimensioned run the bar follows; otherwise omit the weight/length item and propose the "
    "count/diameter/spacing items alone instead (those ARE directly readable from a bar-mark "
    "callout).\n\n"
    "ROLE-NAMING RULE (this is what makes the takeoff actually compute -- read carefully): a "
    "downstream matcher resolves each `depends_on` role by looking for a dimension on THIS drawing "
    "whose own recorded text -- its element description, its section/part label, or its raw printed "
    "label -- literally contains every word in your role name. It does NOT understand synonyms or "
    "meaning, only literal shared words. So:\n"
    "   - Build each role name ONLY from words that plausibly already appear (or would naturally be "
    "transcribed) in this drawing's own annotation text for that exact dimension -- e.g. if the "
    "drawing labels a dimension near a footing as 'Footing Width', propose 'footing_width', not "
    "'foundation_base_width' or 'pad_width'.\n"
    "   - Prefer short, concrete, physically-named roles: '<element>_<measure>' (e.g. "
    "'wall_height', 'footing_depth', 'duct_length', 'slab_thickness') over abstract or bundled "
    "names ('structural_dimension_1', 'main_size').\n"
    "   - Where this project's own dimension-type vocabulary already has a matching category -- "
    "wall_thickness, floor_to_floor, guardrail_height, tread_going, stair_rise, stair_width, "
    "clearance, headroom, level_difference -- reuse that exact word/phrase inside your role name, "
    "since the extraction pass tags matching dimensions with these exact type words.\n"
    "   - For rebar/bar-mark callouts specifically (e.g. a label like 'Y32-100' or '12 Y25'), use "
    "role names ending in exactly '_bar_diameter', '_bar_spacing', '_bar_count', or '_bar_length' "
    "(e.g. 'main_bar_diameter', 'stirrup_bar_spacing') -- a downstream reader parses these specific "
    "sub-values directly out of a bar-mark callout's own text, so do NOT invent a role like "
    "'rebar_size' or 'bar_info' that bundles diameter+spacing together; split them into separate "
    "roles instead.\n"
    "   - If you cannot picture the specific words this drawing would use for a role's own "
    "dimension text, don't guess a generic name and hope -- either skip that item, or (if the value "
    "is something the model itself should read directly, e.g. an odd or unlabeled figure) still "
    "name the role after what's actually printed near it.\n\n"
    "For each item you do propose:\n"
    "   - name: a stable snake_case identifier, e.g. 'foundation_concrete_volume', 'rebar_weight', "
    "'formwork_area', 'duct_length'.\n"
    "   - unit, measurement_basis (count/length/area/volume/weight), resource_category "
    "(material/equipment).\n"
    "   - depends_on: the input roles this formula needs (see ROLE-NAMING RULE above), NOT specific "
    "numbers or this drawing's own dimension IDs.\n"
    "   - input_units: for every role in depends_on, state the unit (mm/cm/m) YOUR formula's own "
    "convention needs it in -- e.g. a rebar weight formula conventionally mixes a millimetre bar "
    "diameter with a metre bar length in one expression, so state 'main_bar_diameter': 'mm' and "
    "'main_bar_length': 'm' explicitly rather than assuming one unit for everything. Whatever unit "
    "you state here is what the value will be converted to before your formula runs, regardless of "
    "what unit this drawing happens to be dimensioned in.\n"
    "   - formula: plain arithmetic over those role names only (+ - * / ** and parentheses) -- e.g. "
    "'footing_length * footing_width * footing_depth'. Never write a computed number as the "
    "formula's result; you are proposing HOW to compute it, not computing it. A formula may ONLY "
    "reference `depends_on` roles that are themselves matchable to something drawn/labeled/"
    "dimensioned on THIS drawing (a length, count, weight-per-length, etc.).\n\n"
    "Respond with ONLY a single raw JSON object -- no markdown code fences, no explanation before or "
    "after it -- with exactly two top-level keys, in this exact shape:\n"
    "{\n"
    '  "classification": {\n'
    '    "project_type": "<string or null>", "discipline": "<string or null>",\n'
    f'    "drawing_type": "<one of {DRAWING_TYPES}>",\n'
    '    "building_elements": ["<string>", "..."],\n'
    f'    "confidence": "<one of {CONFIDENCE_LEVELS}>", "notes": "<string or null>"\n'
    "  },\n"
    '  "quantity_items": [\n'
    "    {\n"
    '      "name": "<snake_case string>", "description": "<string or null>", "unit": "<string>",\n'
    f'      "measurement_basis": "<one of {MEASUREMENT_BASES}>",\n'
    f'      "resource_category": "<one of {RESOURCE_CATEGORIES}>",\n'
    '      "depends_on": ["<role_name>", "..."],\n'
    '      "input_units": {"<role_name>": "<mm|cm|m>", "...": "..."},\n'
    '      "formula": "<arithmetic expression string>"\n'
    "    }\n"
    "  ]\n"
    "}\n"
    "`quantity_items` may be an empty array; `classification` is always required. Your entire reply "
    "must be parseable by a strict JSON parser as-is -- do not wrap it in ```json fences, do not add "
    "commentary, do not truncate it."

    #old
    # "You are an expert estimator reviewing ONE architectural/structural/civil/MEP drawing (already "
    # "cropped from its sheet). Two jobs, in order:\n\n"
    # "1. Classify it: project_type (the kind of facility/project this drawing belongs to), discipline "
    # "(architectural/structural/MEP/civil/etc.), drawing_type, and building_elements (the specific "
    # "elements/systems actually shown -- e.g. ['stair', 'handrail'], ['foundation', 'footing', "
    # "'rebar'], ['duct', 'diffuser'], ['road', 'pavement', 'curb']). Set confidence to 'low' if the "
    # "drawing's subject is genuinely ambiguous (e.g. a fragment with no title or legible content) "
    # "rather than guessing.\n\n"
    # "2. Propose a resource-planning quantity-takeoff schema for THIS ONE DRAWING SPECIFICALLY: "
    # "think like an estimator building a bill of quantities for labor, materials, and equipment. "
    # "This schema is generated fresh for this drawing alone -- it is never shared, reused, or "
    # "extended from another drawing's schema, even one classified identically, because two "
    # "drawings with the same discipline/drawing_type can still show entirely different physical "
    # "subjects (e.g. a full wall section vs. a small local recess detail). Propose ONLY items "
    # "genuinely derivable from what THIS specific drawing itself shows -- an empty quantity_items "
    # "list is correct and expected when nothing meaningful can be derived (e.g. a key plan, a "
    # "legend, a fragment with no legible geometry, or a note-only sheet).\n\n"
    # "CRITICAL -- respect what this drawing's own drawing_type can physically show:\n"
    # "   - A section or elevation is a cut THROUGH a member -- it can show that member's height, "
    # "depth, and thickness (the dimensions visible in the cut plane), and it can show bar size/"
    # "spacing callouts, but it CANNOT show that member's own length along its run (how far a wall "
    # "or slab extends in plan) -- that is only visible in a plan view. Do not propose a role like "
    # "'wall_length' or 'slab_length' on a section/elevation drawing; if a quantity genuinely needs "
    # "a run length, either leave it out of quantity_items entirely, or note in the item's "
    # "description that it requires a companion plan view this drawing does not provide.\n"
    # "   - A rebar weight/length item needs the bar's own physical cut length as an input. A bar's "
    # "size and spacing ARE visible on a section (its bar-mark callout), but its cut length almost "
    # "never is -- do not propose a role for it unless this specific drawing actually shows a "
    # "dimensioned run the bar follows; otherwise omit the weight/length item and propose the "
    # "count/diameter/spacing items alone instead (those ARE directly readable from a bar-mark "
    # "callout).\n"
    # "For each item you do propose:\n"
    # "   - name: a stable snake_case identifier, e.g. 'foundation_concrete_volume', "
    # "'rebar_weight', 'formwork_area', 'duct_length'.\n"
    # "   - unit, measurement_basis (count/length/area/volume/weight), resource_category "
    # "(material/labor/equipment).\n"
    # "   - depends_on: the input roles this formula needs, NOT specific numbers or this drawing's "
    # "own dimension IDs -- clear, descriptive snake_case names (e.g. 'footing_width', "
    # "'wall_height', 'slab_thickness'). For rebar/bar-mark callouts specifically (e.g. a label "
    # "like 'Y32-100' or '12 Y25'), use role names ending in exactly '_bar_diameter', "
    # "'_bar_spacing', '_bar_count', or '_bar_length' (e.g. 'main_bar_diameter', "
    # "'stirrup_bar_spacing') -- a downstream reader parses these specific sub-values directly out "
    # "of a bar-mark callout's own text, so do NOT invent a role like 'rebar_size' or 'bar_info' "
    # "that bundles diameter+spacing together; split them into separate roles instead. Where this "
    # "project's own dimension vocabulary already has a matching category -- wall_thickness, "
    # "floor_to_floor, guardrail_height, tread_going, stair_rise -- reuse that exact word inside "
    # "your role name.\n"
    # "   - input_units: for every role in depends_on, state the unit (mm/cm/m) YOUR formula's own "
    # "convention needs it in -- e.g. a rebar weight formula conventionally mixes a millimetre bar "
    # "diameter with a metre bar length in one expression, so state 'main_bar_diameter': 'mm' and "
    # "'main_bar_length': 'm' explicitly rather than assuming one unit for everything. Whatever unit "
    # "you state here is what the value will be converted to before your formula runs, regardless of "
    # "what unit this drawing happens to be dimensioned in.\n"
    # "   - formula: plain arithmetic over those role names only (+ - * / ** and parentheses) -- e.g. "
    # "'footing_length * footing_width * footing_depth'. Never write a computed number as the "
    # "formula's result; you are proposing HOW to compute it, not computing it. A formula may ONLY "
    # "reference `depends_on` roles that are themselves matchable to something drawn/labeled/"
    # "dimensioned on THIS drawing (a length, count, weight-per-length, etc.) -- NEVER reference an "
    # "external reference constant that no drawing could ever show, such as a labor productivity "
    # "rate, an equipment output rate, or a material unit-weight factor (e.g. "
    # "'placement_rate_per_hour', 'pump_rate_per_hour', 'labor_hours_per_kg', "
    # "'rebar_unit_weight_factor'). If a genuinely useful quantity (like installation labor-hours) "
    # "would require such a constant, leave it out entirely rather than proposing an item that can "
    # "never compute -- propose the underlying material quantity itself instead (e.g. "
    # "'rebar_weight' in kg, not 'rebar_installation_labor' in hours).\n\n"
    # "Respond with ONLY a single raw JSON object -- no markdown code fences, no explanation before "
    # "or after it -- with exactly two top-level keys, in this exact shape:\n"
    # "{\n"
    # '  "classification": {\n'
    # '    "project_type": "<string or null>", "discipline": "<string or null>",\n'
    # f'    "drawing_type": "<one of {DRAWING_TYPES}>",\n'
    # '    "building_elements": ["<string>", "..."],\n'
    # f'    "confidence": "<one of {CONFIDENCE_LEVELS}>", "notes": "<string or null>"\n'
    # "  },\n"
    # '  "quantity_items": [\n'
    # "    {\n"
    # '      "name": "<snake_case string>", "description": "<string or null>", "unit": "<string>",\n'
    # f'      "measurement_basis": "<one of {MEASUREMENT_BASES}>",\n'
    # f'      "resource_category": "<one of {RESOURCE_CATEGORIES}>",\n'
    # '      "depends_on": ["<role_name>", "..."],\n'
    # '      "input_units": {"<role_name>": "<mm|cm|m>", "...": "..."},\n'
    # '      "formula": "<arithmetic expression string>"\n'
    # "    }\n"
    # "  ]\n"
    # "}\n"
    # "`quantity_items` may be an empty array; `classification` is always required. Your entire "
    # "reply must be parseable by a strict JSON parser as-is -- do not wrap it in ```json fences, "
    # "do not add commentary, do not truncate it."
)


def _validate_classification(raw: object) -> dict:
    """Raise a descriptive ValueError if the classification tool call's
    input doesn't match the expected shape."""
    if not isinstance(raw, dict):
        raise ValueError(f"classification input must be a JSON object, got {type(raw).__name__}: {raw!r}")

    classification = raw.get("classification")
    if not isinstance(classification, dict):
        raise ValueError(f"'classification' must be a JSON object, got: {classification!r}")
    if not classification.get("drawing_type"):
        raise ValueError("'classification.drawing_type' is required")

    items = raw.get("quantity_items", [])
    if not isinstance(items, list) or any(not isinstance(i, dict) for i in items):
        raise ValueError(f"'quantity_items' must be an array of JSON objects, got: {items!r}")
    for item in items:
        missing = [
            k for k in ("name", "unit", "measurement_basis", "resource_category", "depends_on", "formula")
            if k not in item
        ]
        if missing:
            raise ValueError(f"quantity_item missing required fields {missing}: {item!r}")
        if not isinstance(item.get("depends_on"), list):
            raise ValueError(f"quantity_item.depends_on must be an array, got: {item.get('depends_on')!r}")
        input_units = item.get("input_units", {})
        if not isinstance(input_units, dict):
            raise ValueError(f"quantity_item.input_units must be an object, got: {input_units!r}")

    return raw


_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)


def _parse_json_response(text: str) -> dict:
    """Parse a plain-text model response as JSON, tolerating the common case
    of the model wrapping it in ```json ... ``` fences despite being told
    not to. Raises ValueError (not json.JSONDecodeError) so callers have one
    exception type to catch, matching _validate_classification's contract.
    """
    stripped = _JSON_FENCE_RE.sub("", text.strip()).strip()
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ValueError(f"response was not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"response JSON must be an object, got {type(parsed).__name__}: {parsed!r}")
    return parsed


def classify_drawing(
    image_bytes: bytes,
    ocr_text: str,
    page_number: int,
    settings: Settings,
    usage_sink: list[LlmCallUsage] | None = None,
    drawing_id: str | None = None,
) -> dict:
    """Call Claude on one drawing's (cropped) image to classify it and
    propose a dynamic quantity-takeoff schema for it.

    Unlike extract_page, this is a PLAIN prompt call -- no tools/tool_choice
    forcing a particular call shape. CLASSIFICATION_SYSTEM_PROMPT itself
    spells out the required JSON shape in text, and the model's plain-text
    reply is parsed as JSON here (_parse_json_response) and validated
    (_validate_classification). This is what lets the proposed schema flow
    into extract_page's own tool schema afterward (see
    _tool_schema_with_dynamic_properties) instead of living behind a second,
    separate forced-tool-call schema.

    Same retry-with-error-feedback pattern as extract_page, just carried by
    a plain user-turn message instead of a tool_result block (there's no
    tool_use to attach one to). Returns {"classification": {...},
    "quantity_items": [...]}; on repeated failure returns an empty-but-valid
    shape (low-confidence 'other' classification, no items) so one bad
    drawing doesn't take down the whole run -- callers should treat a
    low-confidence/'other' result as "no dynamic takeoff for this drawing"
    rather than retrying indefinitely.
    """
    client = _build_client(settings)
    image_b64 = base64.b64encode(image_bytes).decode("ascii")
    messages = _build_messages(image_b64, ocr_text, page_number)

    last_error: Exception | None = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            message = client.messages.create(
                model=settings.foundry_claude_deployment,
                max_tokens=4096,
                system=CLASSIFICATION_SYSTEM_PROMPT,
                messages=messages,
            )
        except Exception as exc:
            logger.warning("Classify page %s attempt %s: request failed: %s", page_number, attempt, exc)
            last_error = exc
            continue

        _record_usage(usage_sink, "classification", page_number, drawing_id, attempt, message, settings)

        raw_text = "".join(block.text for block in message.content if block.type == "text")
        if not raw_text.strip():
            last_error = ValueError("Claude's reply contained no text content")
            logger.warning("Classify page %s attempt %s: %s", page_number, attempt, last_error)
            continue

        try:
            parsed = _parse_json_response(raw_text)
            return _validate_classification(parsed)
        except ValueError as exc:
            logger.warning("Classify page %s attempt %s: malformed JSON reply: %s", page_number, attempt, exc)
            last_error = exc
            messages.append({"role": "assistant", "content": raw_text})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Your last reply was invalid: {exc}. Reply again with ONLY the corrected "
                        "raw JSON object (no markdown fences, no commentary), matching the exact "
                        "shape described in the system prompt: a top-level 'classification' object "
                        "with at least 'drawing_type' and 'confidence', and a top-level "
                        "'quantity_items' array of objects each with name/unit/measurement_basis/"
                        "resource_category/depends_on/formula."
                    ),
                }
            )

    logger.error(
        "Giving up on classifying page %s drawing after %s attempts (%s); "
        "returning a low-confidence 'other' classification with no takeoff items",
        page_number, MAX_ATTEMPTS, last_error,
    )
    return {
        "classification": {
            "drawing_type": "other",
            "confidence": "low",
            "notes": f"classification failed after {MAX_ATTEMPTS} attempts: {last_error}",
        },
        "quantity_items": [],
    }
