"""Computes every drawing's dynamic quantity takeoff: matches each schema
item's declared input roles to that drawing's own extracted dimensions,
then hands the resolved inputs to pipeline.calculations.compute_dynamic_quantity
to actually evaluate the formula. Every drawing goes through this same path
-- there is no longer a specialized per-element-type handler.
"""
from __future__ import annotations

import re

from .calculations import compute_dynamic_quantity
from .schema import ComputedQuantity, Dimension, Drawing, DynamicTakeoff
from .schema_registry import SchemaRegistryEntry


def _role_search_text(dim: Dimension) -> str:
    return " ".join(
        part for part in (dim.type, dim.element, dim.section_part, dim.label_text) if part
    ).lower()


# Rebar bar-mark callouts (the "Y32-100", "12 Y32", "2 Y12-150x500", "2x4 Y16"
# convention widely used on UK/Gulf-standard structural drawings) carry real
# numeric content -- bar diameter, spacing, count, sometimes a length -- but
# arrive as a single text label with `value: null`, since they aren't a
# spanning dimension line. Rather than leaving that numeric content
# unreachable, this parses it out of `label_text` so roles like
# 'bar_diameter'/'bar_spacing'/'bar_count' can still resolve -- purely a
# text-pattern reader over what the model already transcribed, never a
# guess at a number not actually printed.
_REBAR_CALLOUT_RE = re.compile(
    r"(?:(?P<mult_a>\d+)\s*[xX]\s*(?P<mult_b>\d+)\s+)?"
    r"(?:(?P<count>\d+)\s+)?"
    r"Y(?P<diameter>\d+)"
    r"(?:-(?P<spacing>\d+))?"
    r"(?:[xX](?P<length>\d+))?",
    re.IGNORECASE,
)

_REBAR_ROLE_FIELDS = {
    "diameter": "diameter",
    "spacing": "spacing",
    "count": "count",
    "length": "length",
}


def _parse_rebar_callout(label_text: str) -> dict[str, float]:
    """Extract whichever of {diameter, spacing, count, length} a rebar
    bar-mark callout's label_text carries, or {} if it doesn't look like
    one at all."""
    match = _REBAR_CALLOUT_RE.search(label_text or "")
    if not match:
        return {}
    fields: dict[str, float] = {"diameter": float(match.group("diameter"))}
    if match.group("spacing"):
        fields["spacing"] = float(match.group("spacing"))
    if match.group("length"):
        fields["length"] = float(match.group("length"))
    if match.group("mult_a") and match.group("mult_b"):
        fields["count"] = float(match.group("mult_a")) * float(match.group("mult_b"))
    elif match.group("count"):
        fields["count"] = float(match.group("count"))
    return fields


def _rebar_field_for_role(role_words: list[str]) -> str | None:
    """Which rebar sub-value a role is asking for (its last word, if it
    names one of diameter/spacing/count/length), or None if this role
    isn't a rebar-callout role at all."""
    if not role_words or not any(w in ("bar", "rebar") for w in role_words):
        return None
    return _REBAR_ROLE_FIELDS.get(role_words[-1])


_VERTICAL_ROLE_WORDS = {"height", "depth", "thickness", "vertical", "rise", "drop"}
_HORIZONTAL_ROLE_WORDS = {"length", "width", "span", "run", "horizontal"}


def _match_dimension_for_role(role: str, dimensions: list[Dimension]) -> tuple[float, str] | None:
    """Find the single dimension on this drawing that best matches a
    logical input role (e.g. 'wall_length'), or None if no dimension
    matches, or if more than one plausible match disagrees on value (too
    ambiguous to guess between).

    Three tiers, in order -- the first to produce any candidate wins:
    1. Text match: every word in the role appears together in one
       dimension's type/element/section_part/label_text, using that
       dimension's own numeric `value` -- the original, most precise rule.
    2. Text match + rebar-callout parsing: same text match, but for a
       dimension with no plain `value` (a bar-mark callout like 'Y32-100'),
       parse the specific numeric sub-value ('diameter'/'spacing'/'count'/
       'length') the role is actually asking for out of its label_text.
    3. Orientation fallback: when NEITHER of the above found anything and
       the role's words say it wants a vertical or horizontal quantity
       (e.g. 'wall_height' vs 'wall_length'), match by that dimension's own
       `orientation` instead of by text -- a broader, less precise net used
       only when the precise rules found nothing at all.

    Every tier still refuses to guess when more than one candidate
    disagrees on value.
    """
    role_words = [w for w in re.split(r"[_\s]+", role.strip().lower()) if w]
    if not role_words:
        return None

    rebar_field = _rebar_field_for_role(role_words)
    # The rebar sub-value word ('diameter'/'spacing'/...) and the generic
    # 'bar'/'rebar' word describe WHAT TO EXTRACT, not text printed on the
    # drawing -- only the remaining qualifier words (e.g. 'top', 'main',
    # 'stirrup') need to actually appear in a dimension's own text. This
    # narrower word list is used ONLY to gate which rebar-callout labels
    # count as candidates; ordinary numeric-value matching below still
    # requires the role's full, unmodified word list.
    rebar_search_words = (
        [w for w in role_words if w not in ("bar", "rebar") and w != role_words[-1]]
        if rebar_field is not None
        else []
    )

    text_candidates: list[tuple[float, str]] = []
    rebar_candidates: list[tuple[float, str]] = []
    for dim in dimensions:
        haystack = _role_search_text(dim)
        if dim.value is not None and dim.value > 0:
            if all(word in haystack for word in role_words):
                text_candidates.append((dim.value, dim.dimension_id))
        elif rebar_field is not None and all(word in haystack for word in rebar_search_words):
            parsed = _parse_rebar_callout(dim.label_text)
            if rebar_field in parsed:
                rebar_candidates.append((parsed[rebar_field], dim.dimension_id))

    resolved = _unambiguous(text_candidates) or _unambiguous(rebar_candidates)
    if resolved is not None:
        return resolved

    orientation = None
    if any(w in _VERTICAL_ROLE_WORDS for w in role_words):
        orientation = "vertical"
    elif any(w in _HORIZONTAL_ROLE_WORDS for w in role_words):
        orientation = "horizontal"
    if orientation is None:
        return None

    orientation_candidates = [
        (dim.value, dim.dimension_id)
        for dim in dimensions
        if dim.orientation == orientation and dim.value is not None and dim.value > 0
    ]
    return _unambiguous(orientation_candidates)


def _unambiguous(candidates: list[tuple[float, str]]) -> tuple[float, str] | None:
    """None if there are no candidates, or if more than one candidate
    disagrees on value (too ambiguous to guess between); otherwise the
    (first) candidate -- every caller's contract stays "never guess"."""
    if not candidates:
        return None
    distinct_values = {value for value, _id in candidates}
    if len(distinct_values) > 1:
        return None
    return candidates[0]


def compute_dynamic_takeoff(
    drawing: Drawing,
    entry: SchemaRegistryEntry,
    reported_quantities: dict[str, float] | None = None,
) -> DynamicTakeoff:
    """Compute every quantity item in `entry` for `drawing`.

    `reported_quantities`, when given, is whatever the model reported
    directly under this drawing's merged `quantities` object during its own
    detail-pass extraction (see claude_extractor.extract_page's
    dynamic_properties) -- read at the same time and with the same care as
    every other dimension on the drawing, rather than guessed from free
    text afterward. A role present there is used as-is; only a role NOT
    reported at all falls back to `_match_dimension_for_role`'s dimension-
    text/rebar-callout/orientation heuristics. Either way, an item with any
    unresolved role comes back null with a `reason` (see
    calculations.compute_dynamic_quantity) -- nothing is ever guessed.
    """
    reported_quantities = reported_quantities or {}
    quantities: list[ComputedQuantity] = []
    for item in entry.items:
        resolved: dict[str, tuple[float, str | None]] = {}
        for role in item.depends_on:
            if role in reported_quantities:
                resolved[role] = (reported_quantities[role], f"{drawing.drawing_id}:reported:{role}")
                continue
            match = _match_dimension_for_role(role, drawing.dimensions)
            if match is not None:
                resolved[role] = match
        quantities.append(compute_dynamic_quantity(item, resolved))

    return DynamicTakeoff(schema_id=entry.schema_id, schema_version=entry.version, quantities=quantities)
