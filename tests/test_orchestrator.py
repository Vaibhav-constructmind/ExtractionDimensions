"""Unit tests for the per-drawing stair-quantity-takeoff logic in
pipeline/orchestrator.py: identifying a drawing's own physical flights
(upper/lower, etc.), pairing its own tread/riser/width dimensions, and
computing that drawing's own concrete volumes -- nothing pooled or copied
across drawings in the same stair group.

Run with:
    .venv\\Scripts\\python.exe -m unittest tests.test_orchestrator -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.orchestrator import (
    _assign_stair_quantity_takeoffs,
    _compute_stair_quantity_takeoff_for_drawing,
    _flight_position_from_text,
    _ordered_stair_width_readings,
    _pair_flights_in_drawing,
    _pool_canonical_stair_value,
    _stair_group_key,
    _warn_on_suspect_drawings,
    to_export_dict,
)
from pipeline.schema import Dimension, Drawing, DrawingMetadata, QuantityField, QuantityTakeoff, StepFormula


def _dim(dimension_id, dim_type, count=None, per_step=None, value=None, unit="mm",
         orientation=None, section_part=None, confidence=None):
    step_formula = (
        StepFormula(count=count, riser_or_tread_dim=per_step, calculated_total=(count or 0) * (per_step or 0))
        if count is not None
        else None
    )
    return Dimension(
        dimension_id=dimension_id,
        type=dim_type,
        value=value,
        unit=unit,
        orientation=orientation,
        section_part=section_part,
        confidence=confidence,
        label_text="x",
        step_formula=step_formula,
        page_number=1,
    )


def _drawing(drawing_id, title, dims):
    return Drawing(
        drawing_id=drawing_id,
        page_number=1,
        drawing_type="plan",
        drawing_metadata=DrawingMetadata(drawing_title=title),
        dimensions=dims,
    )


def _takeoff_for_single_drawing(d):
    """Test helper: compute a takeoff for `d` as if it were the only
    drawing in its stair group -- canonical riser/tread values are pooled
    from just this one drawing's own dimensions, matching the old
    single-drawing call site while exercising the real (now group-aware)
    function signature."""
    stair_group_id = _stair_group_key(d.drawing_metadata.drawing_title)
    if stair_group_id is None:
        return None
    canonical_riser = _pool_canonical_stair_value([d], stair_group_id, "stair_rise")
    canonical_tread = _pool_canonical_stair_value([d], stair_group_id, "tread_going")
    return _compute_stair_quantity_takeoff_for_drawing(d, stair_group_id, canonical_riser, canonical_tread)


class TestStairGroupKey(unittest.TestCase):
    def test_variants_resolve_to_same_group(self):
        titles = [
            "STAIR-07-INTERMEDIATE LANDING-03",
            "STAIR-07-BG1-TUNNEL PLAN",
            "STAIR-07-G01-GRADE PLAN",
            "STAIR DETAIL-07",
            "3D_STAIR-07",
        ]
        for title in titles:
            self.assertEqual(_stair_group_key(title), "STAIR-07", msg=title)

    def test_different_stair_number(self):
        self.assertEqual(_stair_group_key("STAIR-12-PLAN"), "STAIR-12")

    def test_no_stair_mention_returns_none(self):
        self.assertIsNone(_stair_group_key("DOOR SCHEDULE"))

    def test_none_title_returns_none(self):
        self.assertIsNone(_stair_group_key(None))


class TestFlightPositionFromText(unittest.TestCase):
    def test_upper_keyword(self):
        self.assertEqual(_flight_position_from_text("Upper flight, left side"), "upper_flight")

    def test_top_keyword_maps_to_upper(self):
        self.assertEqual(_flight_position_from_text("Top flight near grid A"), "upper_flight")

    def test_lower_keyword(self):
        self.assertEqual(_flight_position_from_text("Lower flight width"), "lower_flight")

    def test_bottom_keyword_maps_to_lower(self):
        self.assertEqual(_flight_position_from_text("Bottom-left, below stair"), "lower_flight")

    def test_checks_multiple_texts_in_order(self):
        self.assertEqual(_flight_position_from_text(None, "unclear", "Upper flight width"), "upper_flight")

    def test_no_keyword_returns_none(self):
        self.assertIsNone(_flight_position_from_text("Left side, near grid A", None))

    def test_all_none_returns_none(self):
        self.assertIsNone(_flight_position_from_text(None, None, None))


class TestTreadRiserEqualWhenOneMissing(unittest.TestCase):
    """Dedicated coverage for the explicit rule: if either tread or riser is
    available for a flight on a drawing, use that same value for the
    missing side, rather than leaving the flight uncalculated."""

    def test_tread_missing_uses_riser_value_for_both(self):
        # Both flights share the SAME riser value (165mm) -- no group-wide
        # majority ambiguity, so this isolates the "no tread_going anywhere
        # in the group" self-pairing fallback specifically.
        d = _drawing("P1-D1", "STAIR-07-PLAN", [
            _dim("P1-D1-DIM04", "stair_rise", count=14, per_step=165),
            _dim("P1-D1-DIM05", "stair_rise", count=7, per_step=165),
            _dim("P1-D1-DIM06", "stair_width", value=1570, unit="mm", orientation="vertical"),
        ])
        takeoff = _takeoff_for_single_drawing(d)
        self.assertIsNotNone(takeoff)
        self.assertEqual(len(takeoff.flights), 2)
        by_count = {f.num_steps.value: f for f in takeoff.flights}
        self.assertEqual(by_count[14.0].tread_depth.value, 165)
        self.assertEqual(by_count[14.0].riser_height.value, 165)
        self.assertEqual(by_count[7.0].tread_depth.value, 165)
        self.assertEqual(by_count[7.0].riser_height.value, 165)

    def test_riser_missing_uses_tread_value_for_both(self):
        d = _drawing("P1-D3", "STAIR-07-PLAN", [
            _dim("P1-D3-DIM05", "tread_going", count=14, per_step=290),
            _dim("P1-D3-DIM06", "stair_width", value=1570, unit="mm", orientation="vertical"),
        ])
        takeoff = _takeoff_for_single_drawing(d)
        self.assertIsNotNone(takeoff)
        self.assertEqual(len(takeoff.flights), 1)
        flight = takeoff.flights[0]
        self.assertEqual(flight.tread_depth.value, 290)
        self.assertEqual(flight.riser_height.value, 290)
        self.assertEqual(flight.riser_height.source, "P1-D3-DIM05")

    def test_two_tread_only_flights_each_get_their_own_self_paired_record(self):
        # Matches the real D3 case: two separate 14-tread readings on one
        # drawing, no riser data anywhere -- each becomes its own flight,
        # not collapsed into one.
        d = _drawing("P1-D3", "STAIR-07-PLAN", [
            _dim("P1-D3-DIM05", "tread_going", count=14, per_step=290, section_part="Upper flight"),
            _dim("P1-D3-DIM14", "tread_going", count=14, per_step=290, section_part="Lower flight"),
            _dim("P1-D3-DIM06", "stair_width", value=1570, unit="mm", orientation="vertical"),
        ])
        takeoff = _takeoff_for_single_drawing(d)
        self.assertIsNotNone(takeoff)
        self.assertEqual(len(takeoff.flights), 2)
        labels = sorted(f.flight_label for f in takeoff.flights)
        self.assertEqual(labels, ["lower_flight", "upper_flight"])
        for f in takeoff.flights:
            self.assertEqual(f.riser_height.value, 290)


class TestPairFlightsInDrawing(unittest.TestCase):
    """Your exact example: a drawing with an upper flight (10 treads/risers)
    and a lower flight (14 treads/risers) -- paired within THIS drawing
    only, never against another drawing's dimensions."""

    def test_two_flights_paired_by_matching_count(self):
        d = _drawing("P1-D1", "STAIR-07-PLAN", [
            _dim("P1-D1-DIM01", "tread_going", count=10, per_step=290, section_part="Upper flight"),
            _dim("P1-D1-DIM02", "stair_rise", count=10, per_step=165, section_part="Upper flight"),
            _dim("P1-D1-DIM03", "tread_going", count=14, per_step=290, section_part="Lower flight"),
            _dim("P1-D1-DIM04", "stair_rise", count=14, per_step=165, section_part="Lower flight"),
        ])
        pairs, incomplete = _pair_flights_in_drawing(d)
        self.assertEqual(len(pairs), 2)
        self.assertEqual(incomplete, [])
        counts = sorted(p[0][0] for p in pairs)
        self.assertEqual(counts, [10, 14])

    def test_same_count_appearing_twice_produces_two_separate_pairs(self):
        # Two distinct physical 14-step flights on one drawing (e.g. a
        # section) must NOT be collapsed into one record.
        d = _drawing("P1-D6", "STAIR DETAIL-07", [
            _dim("P1-D6-DIM01", "tread_going", count=14, per_step=290),
            _dim("P1-D6-DIM02", "tread_going", count=14, per_step=290),
            _dim("P1-D6-DIM03", "stair_rise", count=14, per_step=165),
            _dim("P1-D6-DIM04", "stair_rise", count=14, per_step=165),
        ])
        pairs, incomplete = _pair_flights_in_drawing(d)
        self.assertEqual(len(pairs), 2)
        self.assertEqual(incomplete, [])
        # Each pair uses a DIFFERENT dimension_id on each side -- not the same one twice.
        tread_ids = {p[0][2] for p in pairs}
        riser_ids = {p[1][2] for p in pairs}
        self.assertEqual(tread_ids, {"P1-D6-DIM01", "P1-D6-DIM02"})
        self.assertEqual(riser_ids, {"P1-D6-DIM03", "P1-D6-DIM04"})

    def test_riser_only_drawing_self_pairs_using_riser_value_for_tread(self):
        # A section-only drawing with risers but no tread_going -- per the
        # tread/riser-equal-when-one-missing rule, each riser-only reading
        # still resolves to its own flight, using the riser value for both.
        d = _drawing("P1-D6", "STAIR DETAIL-07", [
            _dim("P1-D6-DIM01", "stair_rise", count=15, per_step=175),
            _dim("P1-D6-DIM02", "stair_rise", count=7, per_step=165),
        ])
        pairs, incomplete = _pair_flights_in_drawing(d)
        self.assertEqual(incomplete, [])
        self.assertEqual(len(pairs), 2)
        by_count = {t[0]: (t, r, method) for t, r, method in pairs}
        self.assertEqual(by_count[15][0][1], 175)  # tread_depth == riser value
        self.assertEqual(by_count[15][1][1], 175)
        self.assertIn("tread depth not found", by_count[15][2])
        self.assertEqual(by_count[7][0][1], 165)
        self.assertIn("tread depth not found", by_count[7][2])

    def test_single_unambiguous_leftover_pair_uses_riser_count(self):
        d = _drawing("P1-D1", "STAIR-07-PLAN", [
            _dim("P1-D1-DIM01", "tread_going", count=13, per_step=290),
            _dim("P1-D1-DIM02", "stair_rise", count=14, per_step=165),
        ])
        pairs, incomplete = _pair_flights_in_drawing(d)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(incomplete, [])
        tread, riser, method = pairs[0]
        self.assertEqual(riser[0], 14)
        self.assertIn("disagreed", method)

    def test_multiple_ambiguous_leftovers_each_self_pair_individually(self):
        # More than one leftover on each side is too ambiguous to guess
        # WHICH tread pairs with WHICH riser -- but per the tread/riser-
        # equal rule, each leftover still becomes its own self-paired
        # flight rather than being discarded.
        d = _drawing("P1-D1", "STAIR-07-PLAN", [
            _dim("P1-D1-DIM01", "tread_going", count=9, per_step=290),
            _dim("P1-D1-DIM02", "stair_rise", count=15, per_step=165),
            _dim("P1-D1-DIM03", "stair_rise", count=7, per_step=165),
        ])
        pairs, incomplete = _pair_flights_in_drawing(d)
        self.assertEqual(incomplete, [])
        self.assertEqual(len(pairs), 3)
        counts = sorted(t[0] for t, r, method in pairs)
        self.assertEqual(counts, [7, 9, 15])
        by_count = {t[0]: (t, r) for t, r, method in pairs}
        # The 9-step flight is tread-only -- self-paired using its own tread value.
        self.assertEqual(by_count[9][0][1], 290)
        self.assertEqual(by_count[9][1][1], 290)
        # The 15/7-step flights are riser-only -- self-paired using their own riser value.
        self.assertEqual(by_count[15][0][1], 165)
        self.assertEqual(by_count[7][0][1], 165)


class TestOrderedStairWidthReadings(unittest.TestCase):
    def test_prefers_vertical_over_horizontal(self):
        d = _drawing("P1-D2", "STAIR-07-PLAN", [
            _dim("P1-D2-DIM07", "stair_width", value=1665, unit="mm", orientation="horizontal"),
            _dim("P1-D2-DIM08", "stair_width", value=1570, unit="mm", orientation="vertical"),
        ])
        readings = _ordered_stair_width_readings(d)
        self.assertEqual(len(readings), 1)
        self.assertEqual(readings[0][0], 1570)

    def test_falls_back_to_all_when_none_vertical(self):
        d = _drawing("P1-D2", "STAIR-07-PLAN", [
            _dim("P1-D2-DIM07", "stair_width", value=1665, unit="mm", orientation="horizontal"),
        ])
        readings = _ordered_stair_width_readings(d)
        self.assertEqual(len(readings), 1)
        self.assertEqual(readings[0][0], 1665)

    def test_preserves_order_and_all_vertical_entries(self):
        d = _drawing("P1-D2", "STAIR-07-PLAN", [
            _dim("P1-D2-DIM08", "stair_width", value=1570, unit="mm", orientation="vertical"),
            _dim("P1-D2-DIM13", "stair_width", value=1570, unit="mm", orientation="vertical"),
        ])
        readings = _ordered_stair_width_readings(d)
        self.assertEqual([r[2] for r in readings], ["P1-D2-DIM08", "P1-D2-DIM13"])

    def test_falls_back_to_plain_width_type_when_no_stair_width_tagged(self):
        # The model sometimes mistags the real width dimension as plain
        # "width" instead of "stair_width" -- this must not lose the value.
        d = _drawing("P1-D3", "STAIR-07-PLAN", [
            _dim("P1-D3-DIM06", "width", value=1570, unit="mm", orientation="vertical"),
            _dim("P1-D3-DIM09", "width", value=1570, unit="mm", orientation="horizontal"),
        ])
        readings = _ordered_stair_width_readings(d)
        self.assertEqual(len(readings), 1)
        self.assertEqual(readings[0][0], 1570)
        self.assertEqual(readings[0][2], "P1-D3-DIM06")

    def test_does_not_fall_back_when_a_real_stair_width_exists(self):
        d = _drawing("P1-D5", "STAIR-07-PLAN", [
            _dim("P1-D5-DIM02", "stair_width", value=1220, unit="mm", orientation="vertical"),
            _dim("P1-D5-DIM03", "width", value=1600, unit="mm", orientation="vertical"),
        ])
        readings = _ordered_stair_width_readings(d)
        self.assertEqual(len(readings), 1)
        self.assertEqual(readings[0][0], 1220)


class TestComputeStairQuantityTakeoffForDrawing(unittest.TestCase):
    """Your exact example: D1 has an upper flight (10 steps) and a lower
    flight (14 steps); D2 has upper=13 and lower=13. Each drawing's takeoff
    must be computed from ONLY that drawing's own dimensions."""

    def _d1(self):
        return _drawing("P1-D1", "STAIR-07-PLAN", [
            _dim("P1-D1-DIM01", "tread_going", count=10, per_step=290, section_part="Upper flight"),
            _dim("P1-D1-DIM02", "stair_rise", count=10, per_step=165, section_part="Upper flight"),
            _dim("P1-D1-DIM03", "stair_width", value=1570, unit="mm", orientation="vertical", section_part="Upper flight width"),
            _dim("P1-D1-DIM04", "tread_going", count=14, per_step=290, section_part="Lower flight"),
            _dim("P1-D1-DIM05", "stair_rise", count=14, per_step=165, section_part="Lower flight"),
            _dim("P1-D1-DIM06", "stair_width", value=1570, unit="mm", orientation="vertical", section_part="Lower flight width"),
        ])

    def _d2(self):
        return _drawing("P1-D2", "STAIR-07-PLAN-02", [
            _dim("P1-D2-DIM01", "tread_going", count=13, per_step=290, section_part="Upper flight"),
            _dim("P1-D2-DIM02", "stair_rise", count=13, per_step=165, section_part="Upper flight"),
            _dim("P1-D2-DIM03", "stair_width", value=1665, unit="mm", orientation="vertical", section_part="Upper flight width"),
            _dim("P1-D2-DIM04", "tread_going", count=13, per_step=290, section_part="Lower flight"),
            _dim("P1-D2-DIM05", "stair_rise", count=13, per_step=165, section_part="Lower flight"),
            _dim("P1-D2-DIM06", "stair_width", value=1665, unit="mm", orientation="vertical", section_part="Lower flight width"),
        ])

    def test_d1_has_upper_10_and_lower_14(self):
        takeoff = _takeoff_for_single_drawing(self._d1())
        self.assertIsNotNone(takeoff)
        by_label = {f.flight_label: f for f in takeoff.flights}
        self.assertEqual(set(by_label), {"upper_flight", "lower_flight"})
        self.assertEqual(by_label["upper_flight"].num_steps.value, 10)
        self.assertEqual(by_label["lower_flight"].num_steps.value, 14)

    def test_d2_has_upper_13_and_lower_13_not_d1s_values(self):
        takeoff = _takeoff_for_single_drawing(self._d2())
        self.assertIsNotNone(takeoff)
        by_label = {f.flight_label: f for f in takeoff.flights}
        self.assertEqual(by_label["upper_flight"].num_steps.value, 13)
        self.assertEqual(by_label["lower_flight"].num_steps.value, 13)
        # D2's own stair width (1665), not D1's (1570) -- never borrowed across drawings.
        self.assertEqual(by_label["upper_flight"].stair_width.value, 1665)

    def test_source_drawings_is_just_this_one_drawing(self):
        takeoff = _takeoff_for_single_drawing(self._d1())
        self.assertEqual(takeoff.source_drawings, ["P1-D1"])
        self.assertEqual(takeoff.stair_group_id, "STAIR-07")

    def test_total_volume_is_sum_of_this_drawings_own_flights(self):
        takeoff = _takeoff_for_single_drawing(self._d1())
        expected = sum(f.total_volume.value for f in takeoff.flights)
        self.assertAlmostEqual(takeoff.total_volume.value, expected, places=10)
        self.assertEqual(takeoff.num_steps_total.value, 24)

    def test_no_flight_instance_count_field(self):
        takeoff = _takeoff_for_single_drawing(self._d1())
        for flight in takeoff.flights:
            self.assertFalse(hasattr(flight, "flight_instance_count"))

    def test_non_stair_drawing_returns_none(self):
        d = _drawing("P1-D9", "DOOR SCHEDULE", [
            _dim("P1-D9-DIM01", "tread_going", count=10, per_step=290),
        ])
        self.assertIsNone(_takeoff_for_single_drawing(d))

    def test_landing_with_two_different_widths_per_flight(self):
        # Reproduces STAIR-07-INTERMEDIATE LANDING-03: one drawing, two
        # stacked flights, upper flight is 1600mm wide, lower flight is
        # 1570mm wide -- these must NOT be collapsed into one shared value.
        d = _drawing("P1-D1", "STAIR-07-INTERMEDIATE LANDING-03", [
            _dim("P1-D1-DIM01", "tread_going", count=14, per_step=290, section_part="Upper flight"),
            _dim("P1-D1-DIM02", "stair_rise", count=14, per_step=165, section_part="Upper flight"),
            _dim("P1-D1-DIM17", "stair_width", value=1600, unit="mm", orientation="vertical", section_part="Upper flight clear width"),
            _dim("P1-D1-DIM03", "tread_going", count=7, per_step=290, section_part="Lower flight"),
            _dim("P1-D1-DIM04", "stair_rise", count=7, per_step=165, section_part="Lower flight"),
            _dim("P1-D1-DIM18", "stair_width", value=1570, unit="mm", orientation="vertical", section_part="Lower flight clear width"),
        ])
        takeoff = _takeoff_for_single_drawing(d)
        self.assertIsNotNone(takeoff)
        by_label = {f.flight_label: f for f in takeoff.flights}
        self.assertEqual(by_label["upper_flight"].stair_width.value, 1600)
        self.assertEqual(by_label["lower_flight"].stair_width.value, 1570)

    def test_two_candidates_for_same_flight_prefers_higher_confidence(self):
        # Reproduces the BG1-tunnel-plan case: two stair_width candidates
        # tagged for the SAME flight (upper), one high-confidence (the real
        # wall-to-wall clear width) and one low-confidence (a partial
        # sub-segment) -- the higher-confidence one must win, not whichever
        # was read first.
        d = _drawing("P1-D4", "STAIR-07-BG1-TUNNEL PLAN", [
            _dim("P1-D4-DIM11", "stair_width", value=1375, orientation="vertical",
                 section_part="Upper flight", confidence="low"),
            _dim("P1-D4-DIM12", "stair_width", value=1570, orientation="vertical",
                 section_part="Upper flight", confidence="high"),
            _dim("P1-D4-DIM01", "tread_going", count=10, per_step=290, section_part="Upper flight"),
            _dim("P1-D4-DIM02", "stair_rise", count=10, per_step=165, section_part="Upper flight"),
            _dim("P1-D4-DIM15", "stair_width", value=1570, orientation="vertical",
                 section_part="Lower flight", confidence="high"),
            _dim("P1-D4-DIM03", "tread_going", count=14, per_step=290, section_part="Lower flight"),
            _dim("P1-D4-DIM04", "stair_rise", count=14, per_step=165, section_part="Lower flight"),
        ])
        takeoff = _takeoff_for_single_drawing(d)
        self.assertIsNotNone(takeoff)
        by_label = {f.flight_label: f for f in takeoff.flights}
        self.assertEqual(by_label["upper_flight"].stair_width.value, 1570)
        self.assertEqual(by_label["upper_flight"].stair_width.source, "P1-D4-DIM12")
        self.assertEqual(by_label["lower_flight"].stair_width.value, 1570)

    def test_landing_with_two_widths_but_no_position_text_falls_back_positional(self):
        # No section_part position hints on the width readings -- falls
        # back to positional (appearance-order) assignment rather than
        # crashing or dropping a flight.
        d = _drawing("P1-D1", "STAIR-07-PLAN", [
            _dim("P1-D1-DIM01", "tread_going", count=14, per_step=290, section_part="Upper flight"),
            _dim("P1-D1-DIM02", "stair_rise", count=14, per_step=165, section_part="Upper flight"),
            _dim("P1-D1-DIM03", "tread_going", count=7, per_step=290, section_part="Lower flight"),
            _dim("P1-D1-DIM04", "stair_rise", count=7, per_step=165, section_part="Lower flight"),
            _dim("P1-D1-DIM17", "stair_width", value=1600, unit="mm", orientation="vertical"),
            _dim("P1-D1-DIM18", "stair_width", value=1570, unit="mm", orientation="vertical"),
        ])
        takeoff = _takeoff_for_single_drawing(d)
        self.assertIsNotNone(takeoff)
        by_label = {f.flight_label: f for f in takeoff.flights}
        self.assertEqual(by_label["upper_flight"].stair_width.value, 1600)
        self.assertEqual(by_label["lower_flight"].stair_width.value, 1570)

    def test_no_width_on_drawing_returns_none(self):
        d = _drawing("P1-D1", "STAIR-07-PLAN", [
            _dim("P1-D1-DIM01", "tread_going", count=10, per_step=290),
            _dim("P1-D1-DIM02", "stair_rise", count=10, per_step=165),
        ])
        self.assertIsNone(_takeoff_for_single_drawing(d))

    def test_riser_only_drawing_self_pairs_tread_from_riser(self):
        # This is the real-world section-only case: risers with no tread_going
        # on the same drawing. Per the tread/riser-equal-when-one-missing
        # rule, this still resolves -- each riser reading becomes its own
        # flight, using the riser value for tread depth too. Both flights
        # share the same riser value here so there's no group-wide majority
        # ambiguity -- that's covered separately in TestCanonicalStairValues.
        d = _drawing("P1-D6", "STAIR DETAIL-07", [
            _dim("P1-D6-DIM01", "stair_rise", count=15, per_step=165),
            _dim("P1-D6-DIM02", "stair_rise", count=7, per_step=165),
            _dim("P1-D6-DIM03", "stair_width", value=1570, unit="mm", orientation="vertical"),
        ])
        takeoff = _takeoff_for_single_drawing(d)
        self.assertIsNotNone(takeoff)
        self.assertEqual(len(takeoff.flights), 2)
        by_count = {f.num_steps.value: f for f in takeoff.flights}
        self.assertEqual(by_count[15.0].tread_depth.value, 165)
        self.assertEqual(by_count[15.0].riser_height.value, 165)
        self.assertEqual(by_count[7.0].tread_depth.value, 165)
        self.assertIn("tread depth not found", by_count[15.0].num_steps_method)

    def test_single_shared_width_applies_to_every_flight(self):
        d = _drawing("P1-D1", "STAIR-07-PLAN", [
            _dim("P1-D1-DIM01", "tread_going", count=10, per_step=290),
            _dim("P1-D1-DIM02", "stair_rise", count=10, per_step=165),
            _dim("P1-D1-DIM03", "tread_going", count=14, per_step=290),
            _dim("P1-D1-DIM04", "stair_rise", count=14, per_step=165),
            _dim("P1-D1-DIM05", "stair_width", value=1570, unit="mm", orientation="vertical"),
        ])
        takeoff = _takeoff_for_single_drawing(d)
        self.assertEqual(len(takeoff.flights), 2)
        for f in takeoff.flights:
            self.assertEqual(f.stair_width.value, 1570)
            self.assertEqual(f.stair_width.source, "P1-D1-DIM05")

    def test_flight_beyond_available_widths_is_incomplete(self):
        d = _drawing("P1-D1", "STAIR-07-PLAN", [
            _dim("P1-D1-DIM01", "tread_going", count=10, per_step=290),
            _dim("P1-D1-DIM02", "stair_rise", count=10, per_step=165),
            _dim("P1-D1-DIM03", "tread_going", count=14, per_step=290),
            _dim("P1-D1-DIM04", "stair_rise", count=14, per_step=165),
            _dim("P1-D1-DIM05", "stair_width", value=1570, unit="mm", orientation="vertical"),
            _dim("P1-D1-DIM06", "stair_width", value=1600, unit="mm", orientation="vertical"),
        ])
        takeoff = _takeoff_for_single_drawing(d)
        self.assertEqual(len(takeoff.flights), 2)
        # Two distinct widths, positionally assigned -- not shared.
        widths = sorted(f.stair_width.value for f in takeoff.flights)
        self.assertEqual(widths, [1570, 1600])

    def test_labels_use_upper_lower_when_only_two_flights_and_no_text_hint(self):
        d = _drawing("P1-D1", "STAIR-07-PLAN", [
            _dim("P1-D1-DIM01", "tread_going", count=10, per_step=290),
            _dim("P1-D1-DIM02", "stair_rise", count=10, per_step=165),
            _dim("P1-D1-DIM03", "tread_going", count=14, per_step=290),
            _dim("P1-D1-DIM04", "stair_rise", count=14, per_step=165),
            _dim("P1-D1-DIM05", "stair_width", value=1570, unit="mm", orientation="vertical"),
        ])
        takeoff = _takeoff_for_single_drawing(d)
        labels = [f.flight_label for f in takeoff.flights]
        self.assertEqual(labels, ["upper_flight", "lower_flight"])

    def test_single_flight_label(self):
        d = _drawing("P1-D1", "STAIR-07-PLAN", [
            _dim("P1-D1-DIM01", "tread_going", count=10, per_step=290),
            _dim("P1-D1-DIM02", "stair_rise", count=10, per_step=165),
            _dim("P1-D1-DIM03", "stair_width", value=1570, unit="mm", orientation="vertical"),
        ])
        takeoff = _takeoff_for_single_drawing(d)
        self.assertEqual(takeoff.flights[0].flight_label, "flight")

    def test_three_or_more_flights_use_ordinal_labels(self):
        d = _drawing("P1-D6", "STAIR DETAIL-07", [
            _dim("P1-D6-DIM01", "tread_going", count=10, per_step=290),
            _dim("P1-D6-DIM02", "stair_rise", count=10, per_step=165),
            _dim("P1-D6-DIM03", "tread_going", count=14, per_step=290),
            _dim("P1-D6-DIM04", "stair_rise", count=14, per_step=165),
            _dim("P1-D6-DIM05", "tread_going", count=7, per_step=290),
            _dim("P1-D6-DIM06", "stair_rise", count=7, per_step=165),
            _dim("P1-D6-DIM07", "stair_width", value=1570, unit="mm", orientation="vertical"),
        ])
        takeoff = _takeoff_for_single_drawing(d)
        labels = [f.flight_label for f in takeoff.flights]
        self.assertEqual(labels, ["flight_1", "flight_2", "flight_3"])


class TestPoolCanonicalStairValue(unittest.TestCase):
    """Direct coverage of _pool_canonical_stair_value: majority riser/tread
    value across every drawing in a stair group, with isolated/conflicting
    readings outvoted rather than fabricated away."""

    def test_majority_wins_over_isolated_outlier(self):
        # Your exact example: 165mm risers repeated, one stray 280mm.
        d1 = _drawing("P1-D1", "STAIR-07-PLAN", [
            _dim("P1-D1-DIM01", "stair_rise", count=14, per_step=280),
        ])
        d2 = _drawing("P1-D2", "STAIR-07-PLAN-02", [
            _dim("P1-D2-DIM01", "stair_rise", count=7, per_step=165),
        ])
        d3 = _drawing("P1-D3", "STAIR-07-PLAN-03", [
            _dim("P1-D3-DIM01", "stair_rise", count=14, per_step=165),
            _dim("P1-D3-DIM02", "stair_rise", count=10, per_step=165),
        ])
        result = _pool_canonical_stair_value([d1, d2, d3], "STAIR-07", "stair_rise")
        self.assertIsNotNone(result)
        value, unit, source = result
        self.assertEqual(value, 165)

    def test_majority_tread_going_across_drawings(self):
        d1 = _drawing("P1-D2", "STAIR-07-PLAN", [
            _dim("P1-D2-DIM01", "tread_going", count=13, per_step=290),
        ])
        d2 = _drawing("P1-D3", "STAIR-07-PLAN-02", [
            _dim("P1-D3-DIM01", "tread_going", count=14, per_step=290),
            _dim("P1-D3-DIM02", "tread_going", count=14, per_step=290),
        ])
        d3 = _drawing("P1-D4", "STAIR-07-PLAN-03", [
            _dim("P1-D4-DIM01", "tread_going", count=10, per_step=285),  # isolated outlier
        ])
        result = _pool_canonical_stair_value([d1, d2, d3], "STAIR-07", "tread_going")
        self.assertEqual(result[0], 290)

    def test_ignores_drawings_outside_the_stair_group(self):
        target = _drawing("P1-D1", "STAIR-07-PLAN", [
            _dim("P1-D1-DIM01", "stair_rise", count=10, per_step=165),
        ])
        other_stair = _drawing("P1-D9", "STAIR-12-PLAN", [
            _dim("P1-D9-DIM01", "stair_rise", count=10, per_step=999),
        ])
        result = _pool_canonical_stair_value([target, other_stair], "STAIR-07", "stair_rise")
        self.assertEqual(result[0], 165)

    def test_label_text_fallback_when_step_formula_missing(self):
        # Real extraction gap: the model read the label correctly ("13
        # TREADS X 290mm") but left step_formula empty -- still valid evidence.
        d = _drawing("P1-D2", "STAIR-07-PLAN", [
            Dimension(
                dimension_id="P1-D2-DIM02", type="tread_going", value=290.0, unit="mm",
                label_text="13 TREADS X 290mm", step_formula=None, page_number=1,
            ),
        ])
        result = _pool_canonical_stair_value([d], "STAIR-07", "tread_going")
        self.assertIsNotNone(result)
        self.assertEqual(result[0], 290)
        self.assertEqual(result[2], "P1-D2-DIM02")

    def test_total_only_dimension_is_not_mistaken_for_a_per_step_value(self):
        # A companion "2625" (the flight's TOTAL rise) must NOT be treated
        # as a per-step riser height just because it's tagged stair_rise --
        # its label_text doesn't match the 'N RISERS x Xmm' pattern.
        d = _drawing("P1-D6", "STAIR DETAIL-07", [
            Dimension(
                dimension_id="P1-D6-DIM15", type="stair_rise", value=2625.0, unit="mm",
                label_text="2625", step_formula=None, page_number=1,
            ),
            _dim("P1-D6-DIM14", "stair_rise", count=15, per_step=165),
        ])
        result = _pool_canonical_stair_value([d], "STAIR-07", "stair_rise")
        self.assertEqual(result[0], 165)  # not 2625

    def test_nothing_found_returns_none(self):
        d = _drawing("P1-D1", "STAIR-07-PLAN", [])
        self.assertIsNone(_pool_canonical_stair_value([d], "STAIR-07", "stair_rise"))


class TestCanonicalOverrideEndToEnd(unittest.TestCase):
    """End-to-end: _assign_stair_quantity_takeoffs applies the stair-wide
    canonical riser/tread values to every drawing's flights, not just the
    per-drawing reading -- your exact scenario (165mm repeated, 280mm
    isolated) resolved across a whole stair group."""

    def test_isolated_280mm_riser_is_overridden_by_165mm_majority(self):
        # D1: the outlier (280mm). D2/D3: repeated 165mm. All in one group.
        d1 = _drawing("P1-D1", "STAIR-07-PLAN-A", [
            _dim("P1-D1-DIM01", "stair_rise", count=14, per_step=280),
            _dim("P1-D1-DIM02", "stair_width", value=1570, unit="mm", orientation="vertical"),
        ])
        d2 = _drawing("P1-D2", "STAIR-07-PLAN-B", [
            _dim("P1-D2-DIM01", "stair_rise", count=7, per_step=165),
            _dim("P1-D2-DIM02", "stair_width", value=1570, unit="mm", orientation="vertical"),
        ])
        d3 = _drawing("P1-D3", "STAIR-07-PLAN-C", [
            _dim("P1-D3-DIM01", "stair_rise", count=10, per_step=165),
            _dim("P1-D3-DIM02", "stair_width", value=1570, unit="mm", orientation="vertical"),
        ])
        page_drawings = [d1, d2, d3]
        _assign_stair_quantity_takeoffs(page_drawings)

        # D1's own flight used a canonical (165), not its own local reading (280).
        self.assertIsNotNone(d1.stair_quantity_takeoff)
        d1_flight = d1.stair_quantity_takeoff.flights[0]
        self.assertEqual(d1_flight.riser_height.value, 165)
        self.assertNotEqual(d1_flight.riser_height.value, 280)
        # Source traces to one of the majority readings, not D1's own dimension.
        self.assertIn(d1_flight.riser_height.source, ("P1-D2-DIM01", "P1-D3-DIM01"))

        # D2 and D3 (already 165) are unaffected.
        self.assertEqual(d2.stair_quantity_takeoff.flights[0].riser_height.value, 165)
        self.assertEqual(d3.stair_quantity_takeoff.flights[0].riser_height.value, 165)


class TestStairWidthIsNeverPooledAcrossViews(unittest.TestCase):
    """Your exact STAIR-07 scenario: BG1-tunnel and the three intermediate-
    landing plans all show 1570mm, while the G01-grade-plan genuinely shows
    1600mm for a different condition. Unlike riser height and tread going,
    stair_width must stay per-drawing -- neither value should leak into the
    other drawing's takeoff."""

    def _member(self, drawing_id, title, width_value):
        return _drawing(drawing_id, title, [
            _dim(f"{drawing_id}-DIM01", "tread_going", count=10, per_step=290),
            _dim(f"{drawing_id}-DIM02", "stair_rise", count=10, per_step=165),
            _dim(f"{drawing_id}-DIM03", "stair_width", value=width_value, unit="mm", orientation="vertical"),
        ])

    def test_1570_views_and_1600_grade_plan_stay_independent(self):
        tunnel = self._member("P1-D1", "STAIR-07-BG1-TUNNEL PLAN", 1570)
        landing01 = self._member("P1-D2", "STAIR-07-INTERMEDIATE LANDING-01", 1570)
        landing02 = self._member("P1-D3", "STAIR-07-INTERMEDIATE LANDING-02", 1570)
        landing03 = self._member("P1-D4", "STAIR-07-INTERMEDIATE LANDING-03", 1570)
        grade_plan = self._member("P1-D5", "STAIR-07-G01-GRADE PLAN", 1600)

        page_drawings = [tunnel, landing01, landing02, landing03, grade_plan]
        _assign_stair_quantity_takeoffs(page_drawings)

        for d in (tunnel, landing01, landing02, landing03):
            self.assertIsNotNone(d.stair_quantity_takeoff)
            self.assertEqual(d.stair_quantity_takeoff.flights[0].stair_width.value, 1570)

        # The grade plan keeps its own 1600mm -- not overridden by the 1570mm majority.
        self.assertIsNotNone(grade_plan.stair_quantity_takeoff)
        self.assertEqual(grade_plan.stair_quantity_takeoff.flights[0].stair_width.value, 1600)
        self.assertEqual(grade_plan.stair_quantity_takeoff.flights[0].stair_width.source, "P1-D5-DIM03")

        # Meanwhile riser height (165mm, unanimous here) IS shared group-wide,
        # confirming width and riser/tread are handled by genuinely different rules.
        for d in page_drawings:
            self.assertEqual(d.stair_quantity_takeoff.flights[0].riser_height.value, 165)


class TestAssignStairQuantityTakeoffs(unittest.TestCase):
    def test_each_drawing_gets_its_own_independent_takeoff(self):
        d1 = _drawing("P1-D1", "STAIR-07-PLAN", [
            _dim("P1-D1-DIM01", "tread_going", count=10, per_step=290),
            _dim("P1-D1-DIM02", "stair_rise", count=10, per_step=165),
            _dim("P1-D1-DIM03", "stair_width", value=1570, unit="mm", orientation="vertical"),
        ])
        d2 = _drawing("P1-D2", "STAIR-07-PLAN-02", [
            _dim("P1-D2-DIM01", "tread_going", count=13, per_step=290),
            _dim("P1-D2-DIM02", "stair_rise", count=13, per_step=165),
            _dim("P1-D2-DIM03", "stair_width", value=1665, unit="mm", orientation="vertical"),
        ])
        _assign_stair_quantity_takeoffs([d1, d2])

        self.assertIsNotNone(d1.stair_quantity_takeoff)
        self.assertIsNotNone(d2.stair_quantity_takeoff)
        # Independently-computed objects -- never the same shared object,
        # and never carrying each other's step counts.
        self.assertIsNot(d1.stair_quantity_takeoff, d2.stair_quantity_takeoff)
        self.assertEqual(d1.stair_quantity_takeoff.flights[0].num_steps.value, 10)
        self.assertEqual(d2.stair_quantity_takeoff.flights[0].num_steps.value, 13)

    def test_drawing_with_nothing_resolvable_gets_none(self):
        d1 = _drawing("P1-D1", "STAIR-07-PLAN", [
            _dim("P1-D1-DIM01", "tread_going", count=10, per_step=290),
        ])
        _assign_stair_quantity_takeoffs([d1])
        self.assertIsNone(d1.stair_quantity_takeoff)


class TestDynamicTotalVolumeFieldName(unittest.TestCase):
    def test_export_dict_renames_total_volume_key(self):
        d1 = _drawing("P1-D1", "STAIR-07-PLAN", [
            _dim("P1-D1-DIM01", "tread_going", count=10, per_step=290),
            _dim("P1-D1-DIM02", "stair_rise", count=10, per_step=165),
            _dim("P1-D1-DIM03", "stair_width", value=1570, unit="mm", orientation="vertical"),
        ])
        _assign_stair_quantity_takeoffs([d1])

        exported = to_export_dict(d1)
        self.assertIn("total_volume_10_steps_D1", exported["stair_quantity_takeoff"])
        self.assertNotIn("total_volume", exported["stair_quantity_takeoff"])
        self.assertIsNotNone(d1.stair_quantity_takeoff.total_volume)


class TestRiserTreadCountMismatchWarning(unittest.TestCase):
    """Verifies that a drawing's own num_risers_total and num_treads_total
    (the model's own quantity_takeoff fields) are checked for agreement."""

    def _drawing_with_takeoff(self, risers, treads):
        return _drawing("P1-D1", "STAIR-07-PLAN", []).model_copy(update={
            "quantity_takeoff": QuantityTakeoff(
                num_risers_total=QuantityField(value=risers, unit="count") if risers is not None else None,
                num_treads_total=QuantityField(value=treads, unit="count") if treads is not None else None,
            )
        })

    def test_mismatch_logs_a_warning(self):
        d1 = self._drawing_with_takeoff(risers=21, treads=19)
        with self.assertLogs("pipeline.orchestrator", level="WARNING") as cm:
            _warn_on_suspect_drawings([d1])
        self.assertTrue(any("num_risers_total" in msg and "num_treads_total" in msg for msg in cm.output))

    def test_agreement_logs_nothing_about_riser_tread_counts(self):
        d1 = self._drawing_with_takeoff(risers=14, treads=14)
        try:
            with self.assertLogs("pipeline.orchestrator", level="WARNING") as cm:
                _warn_on_suspect_drawings([d1])
            messages = cm.output
        except AssertionError:
            messages = []  # no warnings logged at all -- also a pass
        self.assertFalse(any("num_risers_total" in msg and "num_treads_total" in msg for msg in messages))


if __name__ == "__main__":
    unittest.main()
