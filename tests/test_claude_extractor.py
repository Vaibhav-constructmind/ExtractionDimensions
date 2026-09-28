"""Unit tests for pipeline/claude_extractor.py's non-network pieces: the
plain-JSON response parser used by classify_drawing (no more forced tool
call for classification), the dynamic-schema merge helper used by
extract_page, and classification validation.

Run with:
    .venv\\Scripts\\python.exe -m unittest tests.test_claude_extractor -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.claude_extractor import (
    TOOL_SCHEMA,
    _parse_json_response,
    _tool_schema_with_dynamic_properties,
    _validate_classification,
)


class TestParseJsonResponse(unittest.TestCase):
    def test_plain_json_object(self):
        result = _parse_json_response('{"classification": {"drawing_type": "plan"}}')
        self.assertEqual(result, {"classification": {"drawing_type": "plan"}})

    def test_strips_json_fenced_code_block(self):
        text = '```json\n{"a": 1}\n```'
        self.assertEqual(_parse_json_response(text), {"a": 1})

    def test_strips_bare_fenced_code_block(self):
        text = '```\n{"a": 1}\n```'
        self.assertEqual(_parse_json_response(text), {"a": 1})

    def test_strips_surrounding_whitespace(self):
        text = '   \n  {"a": 1}   \n  '
        self.assertEqual(_parse_json_response(text), {"a": 1})

    def test_invalid_json_raises_value_error(self):
        with self.assertRaises(ValueError):
            _parse_json_response("{not valid json")

    def test_non_object_json_raises_value_error(self):
        with self.assertRaises(ValueError):
            _parse_json_response("[1, 2, 3]")

    def test_empty_string_raises_value_error(self):
        with self.assertRaises(ValueError):
            _parse_json_response("")


class TestToolSchemaWithDynamicProperties(unittest.TestCase):
    def test_merges_quantities_property_into_drawing_item(self):
        dynamic_properties = {
            "wall_height": {"type": ["number", "null"], "description": "Value for role 'wall_height'"},
        }
        merged = _tool_schema_with_dynamic_properties(dynamic_properties)
        item_properties = merged["input_schema"]["properties"]["drawings"]["items"]["properties"]
        self.assertIn("quantities", item_properties)
        self.assertEqual(
            item_properties["quantities"]["properties"]["wall_height"]["type"], ["number", "null"]
        )

    def test_does_not_mutate_the_original_tool_schema(self):
        dynamic_properties = {"wall_height": {"type": "number"}}
        _tool_schema_with_dynamic_properties(dynamic_properties)
        item_properties = TOOL_SCHEMA["input_schema"]["properties"]["drawings"]["items"]["properties"]
        self.assertNotIn("quantities", item_properties)

    def test_two_calls_do_not_leak_properties_into_each_other(self):
        merged_a = _tool_schema_with_dynamic_properties({"a": {"type": "number"}})
        merged_b = _tool_schema_with_dynamic_properties({"b": {"type": "number"}})
        props_a = merged_a["input_schema"]["properties"]["drawings"]["items"]["properties"]["quantities"]["properties"]
        props_b = merged_b["input_schema"]["properties"]["drawings"]["items"]["properties"]["quantities"]["properties"]
        self.assertEqual(set(props_a), {"a"})
        self.assertEqual(set(props_b), {"b"})


class TestValidateClassification(unittest.TestCase):
    def test_valid_shape_passes_through(self):
        raw = {
            "classification": {"drawing_type": "plan", "confidence": "high"},
            "quantity_items": [
                {
                    "name": "x", "unit": "m3", "measurement_basis": "volume",
                    "resource_category": "material", "depends_on": ["a", "b"], "formula": "a * b",
                }
            ],
        }
        self.assertEqual(_validate_classification(raw), raw)

    def test_missing_classification_raises(self):
        with self.assertRaises(ValueError):
            _validate_classification({"quantity_items": []})

    def test_classification_missing_drawing_type_raises(self):
        with self.assertRaises(ValueError):
            _validate_classification({"classification": {"confidence": "high"}, "quantity_items": []})

    def test_quantity_item_missing_required_field_raises(self):
        raw = {
            "classification": {"drawing_type": "plan", "confidence": "high"},
            "quantity_items": [{"name": "x", "unit": "m3"}],
        }
        with self.assertRaises(ValueError):
            _validate_classification(raw)

    def test_empty_quantity_items_is_valid(self):
        raw = {"classification": {"drawing_type": "other", "confidence": "low"}, "quantity_items": []}
        self.assertEqual(_validate_classification(raw), raw)


if __name__ == "__main__":
    unittest.main()
