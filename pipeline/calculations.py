"""Deterministic (non-LLM) engineering calculations over already-extracted
dimensions. Kept separate from orchestrator.py's extraction/correction logic
since these are pure arithmetic over numbers the model already produced, not
extraction post-processing.

Covers: single-stair-step concrete volume, per-flight concrete volume (a
flight = a run of steps sharing one tread depth and riser height), and the
whole-stair total (the exact sum of each flight's own volume). Landing
volume and total PROJECT concrete volume (across multiple stairs) are still
deferred.
"""
from __future__ import annotations

from .schema import FlightConcrete, MeasuredValue, SingleStepConcrete

_UNIT_TO_METRES = {
    "mm": 0.001,
    "cm": 0.01,
    "m": 1.0,
}


def _to_metres(value: float, unit: str) -> float:
    unit_norm = (unit or "").strip().lower()
    if unit_norm not in _UNIT_TO_METRES:
        raise ValueError(
            f"Unsupported unit for stair-step calculation: {unit!r} "
            f"(supported: {sorted(_UNIT_TO_METRES)})"
        )
    return value * _UNIT_TO_METRES[unit_norm]


def calculate_single_step_volume(
    tread_depth: float | None,
    riser_height: float | None,
    stair_width: float | None,
    unit: str = "mm",
) -> float:
    """Concrete volume of one stair step, treated as a triangular prism:

        V = 0.5 * tread_depth * riser_height * stair_width

    `tread_depth`, `riser_height`, and `stair_width` are all in `unit`
    (mm/cm/m, default mm -- matching how dimensions are normally extracted).
    The result is in cubic metres, not rounded -- rounding is a display
    concern, not a calculation concern.

    Raises ValueError if any input is missing, zero, or negative -- a stair
    step can't have a non-positive dimension, so this is a real input error,
    not a value to silently coerce.
    """
    if tread_depth is None or riser_height is None or stair_width is None:
        raise ValueError(
            "tread_depth, riser_height, and stair_width are all required "
            f"(got tread_depth={tread_depth!r}, riser_height={riser_height!r}, "
            f"stair_width={stair_width!r})"
        )
    if tread_depth <= 0 or riser_height <= 0 or stair_width <= 0:
        raise ValueError(
            "tread_depth, riser_height, and stair_width must all be positive "
            f"(got tread_depth={tread_depth!r}, riser_height={riser_height!r}, "
            f"stair_width={stair_width!r})"
        )

    tread_depth_m = _to_metres(tread_depth, unit)
    riser_height_m = _to_metres(riser_height, unit)
    stair_width_m = _to_metres(stair_width, unit)
    return 0.5 * tread_depth_m * riser_height_m * stair_width_m


def build_single_step_concrete(
    tread_depth: float,
    riser_height: float,
    stair_width: float,
    unit: str = "mm",
    tread_depth_source: str | None = None,
    riser_height_source: str | None = None,
    stair_width_source: str | None = None,
) -> SingleStepConcrete:
    """Build the auditable SingleStepConcrete record: the original inputs
    (in `unit`), the computed volume (m3), and the formula as evaluated in
    metres, so the result is checkable by hand. `*_source` should be the
    dimension_id each input was read from, when known -- left None (never
    fabricated) when the caller can't trace an input back to a specific
    extracted dimension.
    """
    volume_m3 = calculate_single_step_volume(tread_depth, riser_height, stair_width, unit=unit)
    tread_depth_m = _to_metres(tread_depth, unit)
    riser_height_m = _to_metres(riser_height, unit)
    stair_width_m = _to_metres(stair_width, unit)

    return SingleStepConcrete(
        tread_depth=MeasuredValue(value=tread_depth, unit=unit),
        riser_height=MeasuredValue(value=riser_height, unit=unit),
        stair_width=MeasuredValue(value=stair_width, unit=unit),
        volume=MeasuredValue(value=volume_m3, unit="m3"),
        formula=f"0.5 × {tread_depth_m:.3f} × {riser_height_m:.3f} × {stair_width_m:.3f}",
        tread_depth_source=tread_depth_source,
        riser_height_source=riser_height_source,
        stair_width_source=stair_width_source,
    )


def calculate_total_steps_volume(volume_per_step_m3: float, num_steps: float) -> float:
    """Total concrete volume for a run of `num_steps` identical steps:

        total = volume_per_step * num_steps

    `volume_per_step_m3` must already be in cubic metres (e.g. from
    calculate_single_step_volume). Raises ValueError for a missing/non-
    positive step count -- a flight can't have zero or a negative number of
    steps, so this is a real input error, not a value to silently coerce.
    """
    if num_steps is None or num_steps <= 0:
        raise ValueError(f"num_steps must be a positive number (got {num_steps!r})")
    return volume_per_step_m3 * num_steps


def select_mode_value(readings: list[tuple[float, float]]) -> float:
    """Given (value, weight) pairs, return the value with the largest total
    weight, tie-broken by the smaller value (a conservative choice for a
    concrete-volume estimate).

    Used two ways: (1) picking a single value out of several raw dimension
    readings that share a step count, each weighted 1.0; and (2) picking a
    single "representative" tread/riser value across a whole stair's
    resolved flights, each weighted by that flight's own step count (so a
    14-step flight influences the representative value more than a 7-step
    one).
    """
    if not readings:
        raise ValueError("select_mode_value requires at least one reading")
    totals: dict[float, float] = {}
    for value, weight in readings:
        totals[value] = totals.get(value, 0.0) + weight
    return min(totals, key=lambda v: (-totals[v], v))


def build_flight_concrete(
    flight_label: str,
    tread_depth: float,
    riser_height: float,
    stair_width: float,
    num_steps: float,
    unit: str = "mm",
    tread_depth_source: str | None = None,
    riser_height_source: str | None = None,
    stair_width_source: str | None = None,
    num_steps_source: str | None = None,
    num_steps_method: str | None = None,
) -> FlightConcrete:
    """Build the auditable FlightConcrete record for ONE physical flight:
    inputs (in `unit`), volume per step, total volume for this flight's
    steps, and both formulas as evaluated in metres, so every number is
    checkable by hand.
    """
    volume_per_step_m3 = calculate_single_step_volume(tread_depth, riser_height, stair_width, unit=unit)
    total_volume_m3 = calculate_total_steps_volume(volume_per_step_m3, num_steps)
    tread_depth_m = _to_metres(tread_depth, unit)
    riser_height_m = _to_metres(riser_height, unit)
    stair_width_m = _to_metres(stair_width, unit)

    return FlightConcrete(
        flight_label=flight_label,
        num_steps=MeasuredValue(value=num_steps, unit="count", source=num_steps_source),
        num_steps_method=num_steps_method,
        tread_depth=MeasuredValue(value=tread_depth, unit=unit, source=tread_depth_source),
        riser_height=MeasuredValue(value=riser_height, unit=unit, source=riser_height_source),
        stair_width=MeasuredValue(value=stair_width, unit=unit, source=stair_width_source),
        volume_per_step=MeasuredValue(value=volume_per_step_m3, unit="m3"),
        formula_per_step=f"0.5 × {tread_depth_m:.3f} × {riser_height_m:.3f} × {stair_width_m:.3f}",
        total_volume=MeasuredValue(value=total_volume_m3, unit="m3"),
        formula_total=f"{int(round(num_steps))} × {volume_per_step_m3:.8f}",
    )
