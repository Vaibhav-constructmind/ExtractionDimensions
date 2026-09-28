"""Unit tests for pipeline/handlers.py: role-matching + dynamic-takeoff
computation.

Run with:
    .venv\\Scripts\\python.exe -m unittest tests.test_handlers -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.handlers import _match_dimension_for_role, _parse_rebar_callout, compute_dynamic_takeoff
from pipeline.schema import Dimension, Drawing, DrawingMetadata, DynamicQuantityItemSpec
from pipeline.schema_registry import SchemaRegistryEntry


def _dim(dimension_id, dim_type, value, element=None, section_part=None, label_text="x"):
    return Dimension(
        dimension_id=dimension_id, type=dim_type, value=value, unit="mm",
        element=element, section_part=section_part, label_text=label_text, page_number=1,
    )


def _drawing(dims):
    return Drawing(
        drawing_id="P1-D1", page_number=1, drawing_type="plan",
        drawing_metadata=DrawingMetadata(drawing_title="FOOTING PLAN"), dimensions=dims,
    )


class TestComputeDynamicTakeoff(unittest.TestCase):
    def test_resolves_roles_by_substring_match_and_computes(self):
        drawing = _drawing([
            _dim("P1-D1-DIM01", "other", 5000.0, element="footing length"),
            _dim("P1-D1-DIM02", "other", 2000.0, element="footing width"),
            _dim("P1-D1-DIM03", "other", 600.0, element="footing depth"),
        ])
        item = DynamicQuantityItemSpec(
            name="footing_concrete_volume", unit="m3", measurement_basis="volume",
            resource_category="material",
            depends_on=["footing_length", "footing_width", "footing_depth"],
            formula="footing_length * footing_width * footing_depth",
        )
        entry = SchemaRegistryEntry(schema_id="abc", version=1, discipline_key="structural", drawing_type_key="plan", items=[item])
        takeoff = compute_dynamic_takeoff(drawing, entry)

        self.assertEqual(takeoff.schema_id, "abc")
        self.assertEqual(len(takeoff.quantities), 1)
        q = takeoff.quantities[0]
        self.assertIsNotNone(q.value)
        self.assertAlmostEqual(q.value.value, 5000.0 * 2000.0 * 600.0)
        self.assertEqual(set(q.sources), {"P1-D1-DIM01", "P1-D1-DIM02", "P1-D1-DIM03"})

    def test_unmatched_role_produces_null_quantity(self):
        drawing = _drawing([_dim("P1-D1-DIM01", "other", 5000.0, element="footing length")])
        item = DynamicQuantityItemSpec(
            name="footing_concrete_volume", unit="m3", measurement_basis="volume",
            resource_category="material",
            depends_on=["footing_length", "footing_width"],
            formula="footing_length * footing_width",
        )
        entry = SchemaRegistryEntry(schema_id="abc", version=1, discipline_key="structural", drawing_type_key="plan", items=[item])
        takeoff = compute_dynamic_takeoff(drawing, entry)
        self.assertIsNone(takeoff.quantities[0].value)
        self.assertIn("footing_width", takeoff.quantities[0].reason)

    def test_reported_quantities_are_used_directly_without_dimension_matching(self):
        # No dimension on this drawing would text-match these roles at all
        # -- reported_quantities alone must be enough to compute.
        drawing = _drawing([])
        item = DynamicQuantityItemSpec(
            name="footing_concrete_volume", unit="m3", measurement_basis="volume",
            resource_category="material",
            depends_on=["footing_length", "footing_width", "footing_depth"],
            formula="footing_length * footing_width * footing_depth",
        )
        entry = SchemaRegistryEntry(schema_id="abc", version=1, discipline_key="structural", drawing_type_key="plan", items=[item])
        reported = {"footing_length": 5000.0, "footing_width": 2000.0, "footing_depth": 600.0}
        takeoff = compute_dynamic_takeoff(drawing, entry, reported)
        q = takeoff.quantities[0]
        self.assertIsNotNone(q.value)
        self.assertAlmostEqual(q.value.value, 5000.0 * 2000.0 * 600.0)
        self.assertTrue(all("reported" in s for s in q.sources))

    def test_reported_quantities_take_priority_over_dimension_matching(self):
        # A dimension WOULD text-match footing_length (5000), but a directly
        # reported value (10) must win instead of the heuristic match.
        drawing = _drawing([_dim("P1-D1-DIM01", "other", 5000.0, element="footing length")])
        item = DynamicQuantityItemSpec(
            name="x", unit="m", measurement_basis="length", resource_category="material",
            depends_on=["footing_length"], formula="footing_length",
        )
        entry = SchemaRegistryEntry(schema_id="abc", version=1, discipline_key="structural", drawing_type_key="plan", items=[item])
        takeoff = compute_dynamic_takeoff(drawing, entry, {"footing_length": 10.0})
        self.assertAlmostEqual(takeoff.quantities[0].value.value, 10.0)

    def test_reported_quantities_missing_role_falls_back_to_matching(self):
        # 'footing_length' is reported directly; 'footing_width' isn't, so
        # it must still fall back to matching that role's own dimension.
        drawing = _drawing([_dim("P1-D1-DIM01", "other", 2000.0, element="footing width")])
        item = DynamicQuantityItemSpec(
            name="x", unit="m2", measurement_basis="area", resource_category="material",
            depends_on=["footing_length", "footing_width"], formula="footing_length * footing_width",
        )
        entry = SchemaRegistryEntry(schema_id="abc", version=1, discipline_key="structural", drawing_type_key="plan", items=[item])
        takeoff = compute_dynamic_takeoff(drawing, entry, {"footing_length": 5000.0})
        self.assertAlmostEqual(takeoff.quantities[0].value.value, 5000.0 * 2000.0)

    def test_ambiguous_conflicting_matches_refuse_to_guess(self):
        drawing = _drawing([
            _dim("P1-D1-DIM01", "width", 5000.0, element="footing length"),
            _dim("P1-D1-DIM02", "width", 9000.0, element="footing length"),
        ])
        item = DynamicQuantityItemSpec(
            name="x", unit="m", measurement_basis="length", resource_category="material",
            depends_on=["footing_length"], formula="footing_length",
        )
        entry = SchemaRegistryEntry(schema_id="abc", version=1, discipline_key="structural", drawing_type_key="plan", items=[item])
        takeoff = compute_dynamic_takeoff(drawing, entry)
        self.assertIsNone(takeoff.quantities[0].value)


class TestParseRebarCallout(unittest.TestCase):
    def test_plain_diameter_and_spacing(self):
        self.assertEqual(_parse_rebar_callout("Y32-100"), {"diameter": 32.0, "spacing": 100.0})

    def test_leading_count(self):
        self.assertEqual(_parse_rebar_callout("12 Y32"), {"diameter": 32.0, "count": 12.0})

    def test_count_diameter_spacing_and_length(self):
        self.assertEqual(
            _parse_rebar_callout("2 Y12-150x500"),
            {"diameter": 12.0, "spacing": 150.0, "length": 500.0, "count": 2.0},
        )

    def test_multiplied_count(self):
        self.assertEqual(_parse_rebar_callout("2x4 Y16 (TYP.)"), {"diameter": 16.0, "count": 8.0})

    def test_trailing_suffix_ignored(self):
        self.assertEqual(_parse_rebar_callout("Y25-200 ADD."), {"diameter": 25.0, "spacing": 200.0})

    def test_web_suffix_ignored(self):
        self.assertEqual(_parse_rebar_callout("Y12-150/WEB"), {"diameter": 12.0, "spacing": 150.0})

    def test_not_a_rebar_callout_returns_empty(self):
        self.assertEqual(_parse_rebar_callout("4200"), {})
        self.assertEqual(_parse_rebar_callout(""), {})


class TestMatchDimensionForRole(unittest.TestCase):
    def test_plain_text_match_returns_value_unit_and_source(self):
        dims = [_dim("P1-D1-DIM01", "other", 5000.0, element="footing length")]
        result = _match_dimension_for_role("footing_length", dims)
        self.assertEqual(result, (5000.0, "mm", "P1-D1-DIM01"))

    def test_rebar_diameter_role_resolves_from_label_text(self):
        dims = [_dim("P1-D1-DIM01", "other", None, element="Top wall rebar", label_text="Y32-100")]
        result = _match_dimension_for_role("top_bar_diameter", dims)
        self.assertEqual(result, (32.0, "mm", "P1-D1-DIM01"))

    def test_rebar_spacing_role_resolves_from_label_text(self):
        dims = [_dim("P1-D1-DIM01", "other", None, element="Top wall rebar", label_text="Y32-100")]
        result = _match_dimension_for_role("top_bar_spacing", dims)
        self.assertEqual(result, (100.0, "mm", "P1-D1-DIM01"))

    def test_rebar_count_role_has_no_unit(self):
        dims = [_dim("P1-D1-DIM01", "other", None, element="Top wall rebar", label_text="12 Y32")]
        result = _match_dimension_for_role("top_bar_count", dims)
        self.assertEqual(result, (12.0, None, "P1-D1-DIM01"))

    def test_rebar_role_still_requires_text_match_first(self):
        # A bar-mark callout that doesn't mention "bottom" at all must not
        # satisfy a 'bottom_bar_diameter' role just because it's a rebar dim.
        dims = [_dim("P1-D1-DIM01", "other", None, element="Top wall rebar", label_text="Y32-100")]
        result = _match_dimension_for_role("bottom_bar_diameter", dims)
        self.assertIsNone(result)

    def test_conflicting_rebar_values_refuse_to_guess(self):
        dims = [
            _dim("P1-D1-DIM01", "other", None, element="Top wall rebar left", label_text="Y32-100"),
            _dim("P1-D1-DIM02", "other", None, element="Top wall rebar right", label_text="Y25-100"),
        ]
        result = _match_dimension_for_role("top_bar_diameter", dims)
        self.assertIsNone(result)

    def test_no_orientation_only_fallback_leaves_role_unresolved(self):
        # A lone vertical-oriented dimension with NO text relation to "wall"
        # or "height" must NOT be claimed by 'wall_height' just because it's
        # the only vertical value on the drawing (the retired orientation-
        # fallback tier used to do this and produced real false positives).
        dims = [Dimension(
            dimension_id="P1-D1-DIM01", type="other", value=3000.0, unit="mm",
            orientation="vertical", label_text="3000", section_part="Recess width between wall faces",
            page_number=1,
        )]
        result = _match_dimension_for_role("wall_height", dims)
        self.assertIsNone(result)


class TestUnitConversionInDynamicTakeoff(unittest.TestCase):
    def _drawing_with_default_units(self, dims, default_units="mm"):
        return Drawing(
            drawing_id="P1-D1", page_number=1, drawing_type="plan",
            drawing_metadata=DrawingMetadata(drawing_title="FOOTING PLAN", default_units=default_units),
            dimensions=dims,
        )

    def test_matched_mm_dimension_converted_to_declared_metre_unit(self):
        drawing = self._drawing_with_default_units([
            _dim("P1-D1-DIM01", "other", 5000.0, element="wall length"),
        ])
        item = DynamicQuantityItemSpec(
            name="x", unit="m", measurement_basis="length", resource_category="material",
            depends_on=["wall_length"], input_units={"wall_length": "m"}, formula="wall_length",
        )
        entry = SchemaRegistryEntry(schema_id="abc", version=1, discipline_key="structural", drawing_type_key="section", items=[item])
        takeoff = compute_dynamic_takeoff(drawing, entry)
        self.assertAlmostEqual(takeoff.quantities[0].value.value, 5.0)  # 5000mm -> 5m

    def test_mixed_unit_rebar_weight_formula_computes_correctly(self):
        # Classic rebar-weight convention: diameter in mm, length in m,
        # combined in one formula -- proves per-role unit declarations let
        # one formula legitimately mix units.
        drawing = self._drawing_with_default_units([
            _dim("P1-D1-DIM01", "other", None, element="main bar", label_text="Y16-150"),
            _dim("P1-D1-DIM02", "other", 2000.0, element="member length"),
        ])
        item = DynamicQuantityItemSpec(
            name="rebar_weight", unit="kg", measurement_basis="weight", resource_category="material",
            depends_on=["main_bar_diameter", "member_length"],
            input_units={"main_bar_diameter": "mm", "member_length": "m"},
            formula="main_bar_diameter * main_bar_diameter * 0.00617 * member_length",
        )
        entry = SchemaRegistryEntry(schema_id="abc", version=1, discipline_key="structural", drawing_type_key="section", items=[item])
        takeoff = compute_dynamic_takeoff(drawing, entry)
        # diameter stays 16mm (already mm), length converts 2000mm -> 2m
        self.assertAlmostEqual(takeoff.quantities[0].value.value, 16.0 * 16.0 * 0.00617 * 2.0)

    def test_no_declared_input_unit_leaves_value_unconverted(self):
        drawing = self._drawing_with_default_units([_dim("P1-D1-DIM01", "other", 600.0, element="wall thickness")])
        item = DynamicQuantityItemSpec(
            name="x", unit="mm", measurement_basis="length", resource_category="material",
            depends_on=["wall_thickness"], formula="wall_thickness",  # no input_units declared
        )
        entry = SchemaRegistryEntry(schema_id="abc", version=1, discipline_key="structural", drawing_type_key="section", items=[item])
        takeoff = compute_dynamic_takeoff(drawing, entry)
        self.assertAlmostEqual(takeoff.quantities[0].value.value, 600.0)

    def test_reported_quantity_uses_drawing_default_units_before_conversion(self):
        drawing = self._drawing_with_default_units([], default_units="mm")
        item = DynamicQuantityItemSpec(
            name="x", unit="m", measurement_basis="length", resource_category="material",
            depends_on=["wall_length"], input_units={"wall_length": "m"}, formula="wall_length",
        )
        entry = SchemaRegistryEntry(schema_id="abc", version=1, discipline_key="structural", drawing_type_key="section", items=[item])
        takeoff = compute_dynamic_takeoff(drawing, entry, {"wall_length": 5000.0})
        self.assertAlmostEqual(takeoff.quantities[0].value.value, 5.0)

    def test_reported_rebar_diameter_stays_mm_regardless_of_drawing_default_units(self):
        drawing = self._drawing_with_default_units([], default_units="m")
        item = DynamicQuantityItemSpec(
            name="x", unit="mm", measurement_basis="length", resource_category="material",
            depends_on=["main_bar_diameter"], input_units={"main_bar_diameter": "mm"}, formula="main_bar_diameter",
        )
        entry = SchemaRegistryEntry(schema_id="abc", version=1, discipline_key="structural", drawing_type_key="section", items=[item])
        takeoff = compute_dynamic_takeoff(drawing, entry, {"main_bar_diameter": 32.0})
        self.assertAlmostEqual(takeoff.quantities[0].value.value, 32.0)


if __name__ == "__main__":
    unittest.main()
