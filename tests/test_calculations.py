"""Unit tests for the deterministic stair concrete-volume calculations
(pipeline/calculations.py): single-step volume, per-flight volume, the
whole-stair total, and the mode/tie-break value selector. Landing volume and
total PROJECT concrete volume (across stairs) are still not implemented.

Run with:
    .venv\\Scripts\\python.exe -m unittest tests.test_calculations -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.calculations import (
    build_flight_concrete,
    build_single_step_concrete,
    calculate_single_step_volume,
    calculate_total_steps_volume,
    select_mode_value,
)

STAIR_07_EXPECTED_VOLUME_M3 = 0.03756225


class TestCalculateSingleStepVolume(unittest.TestCase):
    """Tests for calculate_single_step_volume: the raw V = 0.5*T*R*W arithmetic."""

    def test_stair_07_mm_test_case(self):
        # Stair 07 fixture: tread=290mm, riser=165mm, width=1570mm.
        volume = calculate_single_step_volume(290, 165, 1570, unit="mm")
        self.assertAlmostEqual(volume, STAIR_07_EXPECTED_VOLUME_M3, places=8)

    def test_default_unit_is_mm(self):
        volume = calculate_single_step_volume(290, 165, 1570)
        self.assertAlmostEqual(volume, STAIR_07_EXPECTED_VOLUME_M3, places=8)

    def test_metre_input_gives_same_result(self):
        # Same physical dimensions, already in metres -- must match the mm case exactly.
        volume = calculate_single_step_volume(0.290, 0.165, 1.570, unit="m")
        self.assertAlmostEqual(volume, STAIR_07_EXPECTED_VOLUME_M3, places=8)

    def test_centimetre_input_gives_same_result(self):
        volume = calculate_single_step_volume(29.0, 16.5, 157.0, unit="cm")
        self.assertAlmostEqual(volume, STAIR_07_EXPECTED_VOLUME_M3, places=8)

    def test_zero_tread_raises(self):
        with self.assertRaises(ValueError):
            calculate_single_step_volume(0, 165, 1570)

    def test_negative_riser_raises(self):
        with self.assertRaises(ValueError):
            calculate_single_step_volume(290, -165, 1570)

    def test_missing_tread_raises(self):
        with self.assertRaises(ValueError):
            calculate_single_step_volume(None, 165, 1570)

    def test_missing_stair_width_raises(self):
        with self.assertRaises(ValueError):
            calculate_single_step_volume(290, 165, None)

    def test_unsupported_unit_raises(self):
        with self.assertRaises(ValueError):
            calculate_single_step_volume(290, 165, 1570, unit="ft")


class TestBuildSingleStepConcrete(unittest.TestCase):
    """Tests for build_single_step_concrete: the schema-shaped, auditable record."""

    def test_stair_07_result_shape_and_value(self):
        result = build_single_step_concrete(290, 165, 1570, unit="mm")

        self.assertEqual(result.tread_depth.value, 290)
        self.assertEqual(result.tread_depth.unit, "mm")
        self.assertEqual(result.riser_height.value, 165)
        self.assertEqual(result.riser_height.unit, "mm")
        self.assertEqual(result.stair_width.value, 1570)
        self.assertEqual(result.stair_width.unit, "mm")

        self.assertAlmostEqual(result.volume.value, STAIR_07_EXPECTED_VOLUME_M3, places=8)
        self.assertEqual(result.volume.unit, "m3")

        self.assertEqual(result.formula, "0.5 × 0.290 × 0.165 × 1.570")

    def test_sources_default_to_none_not_fabricated(self):
        result = build_single_step_concrete(290, 165, 1570)
        self.assertIsNone(result.tread_depth_source)
        self.assertIsNone(result.riser_height_source)
        self.assertIsNone(result.stair_width_source)

    def test_sources_are_preserved_when_given(self):
        result = build_single_step_concrete(
            290, 165, 1570,
            tread_depth_source="P1-D2-DIM03",
            riser_height_source="P1-D6-DIM08",
            stair_width_source="P1-D2-DIM07",
        )
        self.assertEqual(result.tread_depth_source, "P1-D2-DIM03")
        self.assertEqual(result.riser_height_source, "P1-D6-DIM08")
        self.assertEqual(result.stair_width_source, "P1-D2-DIM07")

    def test_invalid_inputs_propagate_as_value_error(self):
        with self.assertRaises(ValueError):
            build_single_step_concrete(0, 165, 1570)


class TestCalculateTotalStepsVolume(unittest.TestCase):
    def test_stair_07_14_steps(self):
        total = calculate_total_steps_volume(STAIR_07_EXPECTED_VOLUME_M3, 14)
        self.assertAlmostEqual(total, STAIR_07_EXPECTED_VOLUME_M3 * 14, places=8)

    def test_zero_steps_raises(self):
        with self.assertRaises(ValueError):
            calculate_total_steps_volume(STAIR_07_EXPECTED_VOLUME_M3, 0)

    def test_missing_steps_raises(self):
        with self.assertRaises(ValueError):
            calculate_total_steps_volume(STAIR_07_EXPECTED_VOLUME_M3, None)


class TestSelectModeValue(unittest.TestCase):
    def test_most_frequent_value_wins(self):
        # D6-style: 165mm used in 5 flights, 175mm used in 1.
        readings = [(165.0, 14), (165.0, 14), (165.0, 14), (165.0, 7), (165.0, 10), (175.0, 15)]
        self.assertEqual(select_mode_value(readings), 165.0)

    def test_tie_break_prefers_smaller_value(self):
        readings = [(165.0, 10), (175.0, 10)]
        self.assertEqual(select_mode_value(readings), 165.0)

    def test_single_reading(self):
        self.assertEqual(select_mode_value([(290.0, 1.0)]), 290.0)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            select_mode_value([])


class TestBuildFlightConcrete(unittest.TestCase):
    """Covers your exact example: an upper flight (10 steps) and a lower
    flight (14 steps) on the same drawing, each with its own
    volume_per_step and total_volume, each a fully independent record."""

    def test_upper_flight_10_steps(self):
        flight = build_flight_concrete(
            flight_label="upper_flight",
            tread_depth=290, riser_height=165, stair_width=1570, num_steps=10,
            tread_depth_source="P1-D1-DIM02", riser_height_source="P1-D1-DIM09",
            stair_width_source="P1-D1-DIM03", num_steps_source="P1-D1-DIM09",
        )
        self.assertAlmostEqual(flight.volume_per_step.value, STAIR_07_EXPECTED_VOLUME_M3, places=8)
        self.assertAlmostEqual(flight.total_volume.value, STAIR_07_EXPECTED_VOLUME_M3 * 10, places=8)
        self.assertEqual(flight.num_steps.value, 10)
        self.assertEqual(flight.flight_label, "upper_flight")

    def test_lower_flight_14_steps_is_independent_of_upper_flight(self):
        upper = build_flight_concrete(
            flight_label="upper_flight",
            tread_depth=290, riser_height=165, stair_width=1570, num_steps=10,
        )
        lower = build_flight_concrete(
            flight_label="lower_flight",
            tread_depth=290, riser_height=165, stair_width=1570, num_steps=14,
        )
        # Same volume_per_step (same tread/riser/width), different totals.
        self.assertAlmostEqual(upper.volume_per_step.value, lower.volume_per_step.value, places=8)
        self.assertNotAlmostEqual(upper.total_volume.value, lower.total_volume.value, places=6)
        self.assertAlmostEqual(lower.total_volume.value, STAIR_07_EXPECTED_VOLUME_M3 * 14, places=8)
        self.assertEqual(lower.flight_label, "lower_flight")

    def test_sources_and_method_preserved(self):
        flight = build_flight_concrete(
            flight_label="lower_flight",
            tread_depth=290, riser_height=165, stair_width=1570, num_steps=14,
            tread_depth_source="P1-D1-DIM03", riser_height_source="P1-D1-DIM09",
            stair_width_source="P1-D1-DIM04", num_steps_source="P1-D1-DIM09",
            num_steps_method="tread count and riser count agree (14 steps)",
        )
        self.assertEqual(flight.tread_depth.source, "P1-D1-DIM03")
        self.assertEqual(flight.riser_height.source, "P1-D1-DIM09")
        self.assertEqual(flight.stair_width.source, "P1-D1-DIM04")
        self.assertEqual(flight.num_steps.source, "P1-D1-DIM09")
        self.assertEqual(flight.num_steps_method, "tread count and riser count agree (14 steps)")


if __name__ == "__main__":
    unittest.main()
