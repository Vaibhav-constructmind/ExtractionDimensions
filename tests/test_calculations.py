"""Unit tests for the generic dynamic-schema formula evaluation in
pipeline/calculations.py: the restricted arithmetic evaluator and the
quantity-computation wrapper around it.

Run with:
    .venv\\Scripts\\python.exe -m unittest tests.test_calculations -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.calculations import (
    FormulaError,
    UnitConversionError,
    compute_dynamic_quantity,
    convert_length_unit,
    evaluate_formula,
)
from pipeline.schema import DynamicQuantityItemSpec


class TestConvertLengthUnit(unittest.TestCase):
    def test_mm_to_metres(self):
        self.assertAlmostEqual(convert_length_unit(5000.0, "mm", "m"), 5.0)

    def test_metres_to_mm(self):
        self.assertAlmostEqual(convert_length_unit(5.0, "m", "mm"), 5000.0)

    def test_cm_to_metres(self):
        self.assertAlmostEqual(convert_length_unit(150.0, "cm", "m"), 1.5)

    def test_same_unit_is_a_no_op(self):
        self.assertEqual(convert_length_unit(42.0, "mm", "mm"), 42.0)

    def test_case_insensitive(self):
        self.assertAlmostEqual(convert_length_unit(1000.0, "MM", "M"), 1.0)

    def test_unrecognized_source_unit_raises(self):
        with self.assertRaises(UnitConversionError):
            convert_length_unit(5.0, "ft", "m")

    def test_unrecognized_target_unit_raises(self):
        with self.assertRaises(UnitConversionError):
            convert_length_unit(5.0, "mm", "ft")


class TestEvaluateFormula(unittest.TestCase):
    def test_basic_arithmetic(self):
        self.assertAlmostEqual(evaluate_formula("a * b * c", {"a": 2.0, "b": 3.0, "c": 4.0}), 24.0)

    def test_parentheses_and_operators(self):
        self.assertAlmostEqual(evaluate_formula("(a + b) / c - 1", {"a": 3.0, "b": 5.0, "c": 4.0}), 1.0)

    def test_unary_minus(self):
        self.assertAlmostEqual(evaluate_formula("-a + b", {"a": 5.0, "b": 3.0}), -2.0)

    def test_power(self):
        self.assertAlmostEqual(evaluate_formula("a ** 2", {"a": 3.0}), 9.0)

    def test_unknown_variable_raises(self):
        with self.assertRaises(FormulaError):
            evaluate_formula("a * unknown_var", {"a": 1.0})

    def test_function_call_rejected(self):
        with self.assertRaises(FormulaError):
            evaluate_formula("__import__('os').system('x')", {})

    def test_attribute_access_rejected(self):
        with self.assertRaises(FormulaError):
            evaluate_formula("a.__class__", {"a": 1.0})

    def test_invalid_syntax_raises(self):
        with self.assertRaises(FormulaError):
            evaluate_formula("a * * b", {"a": 1.0, "b": 2.0})


class TestComputeDynamicQuantity(unittest.TestCase):
    def _spec(self, **overrides):
        defaults = dict(
            name="wall_concrete_volume",
            unit="m3",
            measurement_basis="volume",
            resource_category="material",
            depends_on=["wall_length", "wall_height", "wall_thickness"],
            formula="wall_length * wall_height * wall_thickness",
        )
        defaults.update(overrides)
        return DynamicQuantityItemSpec(**defaults)

    def test_all_inputs_resolved_computes_value_with_sources(self):
        item = self._spec()
        resolved = {
            "wall_length": (10.0, "P1-D1-DIM01"),
            "wall_height": (3.0, "P1-D1-DIM02"),
            "wall_thickness": (0.2, "P1-D1-DIM03"),
        }
        result = compute_dynamic_quantity(item, resolved)
        self.assertIsNotNone(result.value)
        self.assertAlmostEqual(result.value.value, 6.0)
        self.assertEqual(result.value.unit, "m3")
        self.assertEqual(set(result.sources), {"P1-D1-DIM01", "P1-D1-DIM02", "P1-D1-DIM03"})
        self.assertIsNone(result.reason)

    def test_missing_input_produces_null_with_reason(self):
        item = self._spec()
        resolved = {"wall_length": (10.0, "P1-D1-DIM01"), "wall_height": (3.0, "P1-D1-DIM02")}
        result = compute_dynamic_quantity(item, resolved)
        self.assertIsNone(result.value)
        self.assertIn("wall_thickness", result.reason)

    def test_no_inputs_at_all_produces_null(self):
        item = self._spec()
        result = compute_dynamic_quantity(item, {})
        self.assertIsNone(result.value)
        self.assertEqual(result.sources, [])

    def test_bad_formula_produces_null_not_a_crash(self):
        item = self._spec(depends_on=["a"], formula="a +")
        result = compute_dynamic_quantity(item, {"a": (1.0, "P1-D1-DIM01")})
        self.assertIsNone(result.value)
        self.assertIn("could not compute", result.reason)


if __name__ == "__main__":
    unittest.main()
