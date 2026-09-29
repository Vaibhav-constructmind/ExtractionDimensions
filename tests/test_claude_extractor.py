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
    _record_usage,
    _tool_schema_with_dynamic_properties,
    _validate_classification,
)
from pipeline.config import Settings


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


def _settings(model="claude-sonnet-5"):
    return Settings(
        doc_intel_endpoint="https://example.invalid",
        doc_intel_key="key",
        foundry_endpoint="https://example.invalid",
        foundry_api_key="key",
        foundry_claude_deployment=model,
        render_dpi=200,
        detail_render_dpi=600,
        schema_output_dir="./schemas",
    )


class _FakeUsage:
    def __init__(self, input_tokens, output_tokens):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class _FakeMessage:
    def __init__(self, usage=None):
        self.usage = usage


class TestRecordUsage(unittest.TestCase):
    def test_none_sink_is_a_noop(self):
        _record_usage(None, "segmentation", 1, None, 1, _FakeMessage(_FakeUsage(10, 20)), _settings())
        # No exception, nothing to assert -- there is no sink to inspect.

    def test_message_without_usage_is_a_noop(self):
        sink: list = []
        _record_usage(sink, "segmentation", 1, None, 1, _FakeMessage(usage=None), _settings())
        self.assertEqual(sink, [])

    def test_records_tokens_and_estimated_cost(self):
        sink: list = []
        message = _FakeMessage(_FakeUsage(1000, 2000))
        _record_usage(sink, "detail", 3, "P3-D1", 2, message, _settings("claude-sonnet-5"))
        self.assertEqual(len(sink), 1)
        record = sink[0]
        self.assertEqual(record.call_type, "detail")
        self.assertEqual(record.page_number, 3)
        self.assertEqual(record.drawing_id, "P3-D1")
        self.assertEqual(record.attempt, 2)
        self.assertEqual(record.input_tokens, 1000)
        self.assertEqual(record.output_tokens, 2000)
        self.assertAlmostEqual(record.cost_usd, 1000 / 1_000_000 * 3.0 + 2000 / 1_000_000 * 15.0)

    def test_each_attempt_appends_its_own_record(self):
        sink: list = []
        _record_usage(sink, "classification", 1, "P1-D1", 1, _FakeMessage(_FakeUsage(100, 100)), _settings())
        _record_usage(sink, "classification", 1, "P1-D1", 2, _FakeMessage(_FakeUsage(100, 100)), _settings())
        self.assertEqual(len(sink), 2)
        self.assertEqual([r.attempt for r in sink], [1, 2])


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
