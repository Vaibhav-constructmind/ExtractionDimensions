"""Computes every drawing's dynamic quantity takeoff: matches each schema
item's declared input roles to that drawing's own extracted dimensions,
converts each resolved value into the unit its own formula needs (see
DynamicQuantityItemSpec.input_units), then hands the converted inputs to
pipeline.calculations.compute_dynamic_quantity to actually evaluate the
formula. Every drawing goes through this same path -- there is no longer a
specialized per-element-type handler.
"""
from __future__ import annotations

import re

from .calculations import UnitConversionError, compute_dynamic_quantity, convert_length_unit
from .schema import ComputedQuantity, Dimension, Drawing, DynamicTakeoff
from .schema_registry import SchemaRegistryEntry

# A resolved-but-not-yet-unit-converted candidate: (value, unit, source).
# `unit` is None for a dimensionless quantity (a bar count) -- no conversion
# is ever attempted for those, regardless of what a formula's input_units
# might say.
_Candidate = tuple[float, str | None, str]


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

# Bar diameter/spacing/length are conventionally printed and read in
# millimetres on these drawings regardless of the drawing's own overall
# `default_units` (a civil drawing dimensioned in metres still calls out
# rebar as e.g. 'Y32-100', always mm) -- a bar count has no unit at all.
_REBAR_FIELD_UNITS = {"diameter": "mm", "spacing": "mm", "length": "mm", "count": None}


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
    if role_words[-1] not in _REBAR_FIELD_UNITS:
        return None
    return role_words[-1]


def _match_dimension_for_role(role: str, dimensions: list[Dimension]) -> _Candidate | None:
    """Find the single dimension on this drawing that best matches a
    logical input role (e.g. 'wall_length'), or None if no dimension
    matches, or if more than one plausible match disagrees on value (too
    ambiguous to guess between).

    Two tiers, in order -- the first to produce any candidate wins:
    1. Text match: every word in the role appears together in one
       dimension's type/element/section_part/label_text, using that
       dimension's own numeric `value` and `unit`.
    2. Text match + rebar-callout parsing: same qualifier-word text match,
       but for a dimension with no plain `value` (a bar-mark callout like
       'Y32-100'), parse the specific numeric sub-value
       ('diameter'/'spacing'/'count'/'length') the role is actually asking
       for out of its label_text, with the fixed unit that sub-value is
       conventionally printed in (mm, or None for a count).

    There is deliberately no broader orientation-only fallback tier: a lone
    dimension sharing a role's vertical/horizontal orientation but with no
    corroborating text match is not reliable evidence of what it actually
    measures (this produced real false-positive matches in practice), so a
    role with no text-level match at all simply stays unresolved.

    Refuses to guess when more than one candidate disagrees on value.
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

    text_candidates: list[_Candidate] = []
    rebar_candidates: list[_Candidate] = []
    for dim in dimensions:
        haystack = _role_search_text(dim)
        if dim.value is not None and dim.value > 0:
            if all(word in haystack for word in role_words):
                text_candidates.append((dim.value, dim.unit or "mm", dim.dimension_id))
        elif rebar_field is not None and all(word in haystack for word in rebar_search_words):
            parsed = _parse_rebar_callout(dim.label_text)
            if rebar_field in parsed:
                rebar_candidates.append((parsed[rebar_field], _REBAR_FIELD_UNITS[rebar_field], dim.dimension_id))

    return _unambiguous(text_candidates) or _unambiguous(rebar_candidates)


def _unambiguous(candidates: list[_Candidate]) -> _Candidate | None:
    """None if there are no candidates, or if more than one candidate
    disagrees on value (too ambiguous to guess between); otherwise the
    (first) candidate -- every caller's contract stays "never guess"."""
    if not candidates:
        return None
    distinct_values = {value for value, _unit, _id in candidates}
    if len(distinct_values) > 1:
        return None
    return candidates[0]


def _reported_quantity_unit(role: str, default_units: str | None) -> str | None:
    """The unit a directly model-reported role's value should be treated
    as being in: the fixed mm convention for a rebar diameter/spacing/
    length role, None (dimensionless) for a rebar count role, otherwise
    this drawing's own default_units (falling back to 'mm', matching how
    dimensions on this project are extracted when no unit is stated)."""
    role_words = [w for w in re.split(r"[_\s]+", role.strip().lower()) if w]
    rebar_field = _rebar_field_for_role(role_words)
    if rebar_field is not None:
        return _REBAR_FIELD_UNITS[rebar_field]
    return default_units or "mm"


def _convert_candidate(value: float, unit: str | None, target_unit: str | None) -> float | None:
    """Convert `value` from `unit` to `target_unit`, or return it
    unconverted if either is missing/dimensionless (nothing to convert) --
    returns None if a conversion was actually needed but the units involved
    aren't a recognized length unit (rather than silently using the raw,
    wrong-scale value)."""
    if target_unit is None or unit is None or unit == target_unit:
        return value
    try:
        return convert_length_unit(value, unit, target_unit)
    except UnitConversionError:
        return None


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
    text/rebar-callout heuristics.

    Every resolved value (however it was found) is converted into the unit
    that item's own `input_units` states for that role before the formula
    ever runs -- e.g. a wall_length matched from a dimension in mm gets
    converted to metres if the formula's own convention calls for metres,
    while a main_bar_diameter stays in mm if that's what the formula wants.
    A role whose value can't be converted into its stated unit (or wasn't
    resolved at all) leaves that item null with a `reason` -- nothing is
    ever guessed or left silently unconverted.
    """
    reported_quantities = reported_quantities or {}
    default_units = drawing.drawing_metadata.default_units
    quantities: list[ComputedQuantity] = []

    for item in entry.items:
        resolved: dict[str, tuple[float, str | None]] = {}
        for role in item.depends_on:
            if role in reported_quantities:
                value = reported_quantities[role]
                unit = _reported_quantity_unit(role, default_units)
                source = f"{drawing.drawing_id}:reported:{role}"
            else:
                match = _match_dimension_for_role(role, drawing.dimensions)
                if match is None:
                    continue
                value, unit, source = match

            converted = _convert_candidate(value, unit, item.input_units.get(role))
            if converted is not None:
                resolved[role] = (converted, source)

        quantities.append(compute_dynamic_quantity(item, resolved))

    return DynamicTakeoff(schema_id=entry.schema_id, schema_version=entry.version, quantities=quantities)
