"""Unit tests for pipeline/schema_registry.py: JSON Schema assembly + meta-
schema validation, reuse/extend-by-classification, and file/manifest I/O.

Run with:
    .venv\\Scripts\\python.exe -m unittest tests.test_schema_registry -v
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.schema import DrawingClassification, DynamicQuantityItemSpec, SchemaHeader
from pipeline.schema_registry import SchemaRegistry, SchemaValidationError, build_json_schema, validate_schema


def _classification(discipline="structural", drawing_type="plan"):
    return DrawingClassification(discipline=discipline, drawing_type=drawing_type, confidence="high")


def _item(name="footing_concrete_volume", depends_on=None, formula="length * width * depth"):
    return DynamicQuantityItemSpec(
        name=name,
        description="Footing concrete volume",
        unit="m3",
        measurement_basis="volume",
        resource_category="material",
        depends_on=depends_on or ["length", "width", "depth"],
        formula=formula,
    )


class TestBuildJsonSchema(unittest.TestCase):
    def test_produces_valid_draft202012_schema(self):
        schema = build_json_schema("abc123", 1, [_item()])
        validate_schema(schema)  # must not raise
        self.assertEqual(schema["properties"]["quantities"]["properties"]["footing_concrete_volume"]["x-unit"], "m3")

    def test_empty_items_still_valid(self):
        schema = build_json_schema("abc123", 1, [])
        validate_schema(schema)


class TestSchemaRegistryReuse(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.registry = SchemaRegistry(self.tmpdir.name, "20260101_120000")

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_new_classification_creates_version_1(self):
        entry, reused = self.registry.get_or_create(_classification(), [_item()])
        self.assertFalse(reused)
        self.assertEqual(entry.version, 1)

    def test_same_classification_reuses_schema_id(self):
        entry1, _ = self.registry.get_or_create(_classification(), [_item()])
        entry2, reused = self.registry.get_or_create(_classification(), [_item()])
        self.assertTrue(reused)
        self.assertEqual(entry1.schema_id, entry2.schema_id)
        self.assertEqual(entry2.version, 1)  # no new items -- no version bump

    def test_new_item_on_reuse_extends_and_bumps_version(self):
        entry1, _ = self.registry.get_or_create(_classification(), [_item()])
        entry2, reused = self.registry.get_or_create(
            _classification(), [_item(), _item(name="rebar_weight", depends_on=["length"], formula="length * 7.85")]
        )
        self.assertTrue(reused)
        self.assertEqual(entry1.schema_id, entry2.schema_id)
        self.assertEqual(entry2.version, 2)
        self.assertEqual({i.name for i in entry2.items}, {"footing_concrete_volume", "rebar_weight"})

    def test_different_classification_gets_different_schema_id(self):
        entry1, _ = self.registry.get_or_create(_classification(discipline="structural"), [_item()])
        entry2, _ = self.registry.get_or_create(_classification(discipline="MEP"), [_item()])
        self.assertNotEqual(entry1.schema_id, entry2.schema_id)

    def test_classification_key_is_case_and_whitespace_insensitive(self):
        entry1, _ = self.registry.get_or_create(_classification(discipline="Structural"), [_item()])
        entry2, reused = self.registry.get_or_create(_classification(discipline="  structural "), [_item()])
        self.assertTrue(reused)
        self.assertEqual(entry1.schema_id, entry2.schema_id)


class TestSchemaRegistryFileIO(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.registry = SchemaRegistry(self.tmpdir.name, "20260101_120000")

    def tearDown(self):
        self.tmpdir.cleanup()

    def _header(self, classification):
        return SchemaHeader(
            schema_id="", version=0, run_timestamp="20260101_120000",
            source_pdf="test.pdf", page=1, drawing_index=1,
            drawing_title="FOOTING PLAN", classification=classification,
        )

    def test_write_schema_file_creates_file_with_header(self):
        classification = _classification()
        entry, _ = self.registry.get_or_create(classification, [_item()])
        header = self._header(classification).model_copy(update={"schema_id": entry.schema_id, "version": entry.version})
        path = self.registry.write_schema_file_if_needed(entry, header)
        self.assertIsNotNone(path)
        self.assertTrue(Path(path).exists())
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        self.assertEqual(payload["header"]["source_pdf"], "test.pdf")
        self.assertEqual(payload["schema_id"], entry.schema_id)

    def test_write_schema_file_is_noop_when_already_written(self):
        classification = _classification()
        entry, _ = self.registry.get_or_create(classification, [_item()])
        header = self._header(classification).model_copy(update={"schema_id": entry.schema_id, "version": entry.version})
        first = self.registry.write_schema_file_if_needed(entry, header)
        second = self.registry.write_schema_file_if_needed(entry, header)
        self.assertIsNotNone(first)
        self.assertIsNone(second)

    def test_extending_schema_allows_rewrite_for_new_version(self):
        classification = _classification()
        entry, _ = self.registry.get_or_create(classification, [_item()])
        header = self._header(classification).model_copy(update={"schema_id": entry.schema_id, "version": entry.version})
        self.registry.write_schema_file_if_needed(entry, header)

        entry2, _ = self.registry.get_or_create(classification, [_item(name="new_item", depends_on=["length"], formula="length")])
        header2 = header.model_copy(update={"version": entry2.version})
        path2 = self.registry.write_schema_file_if_needed(entry2, header2)
        self.assertIsNotNone(path2)

    def test_manifest_records_every_drawing(self):
        classification = _classification()
        entry, reused = self.registry.get_or_create(classification, [_item()])
        header = self._header(classification).model_copy(update={"schema_id": entry.schema_id, "version": entry.version})
        self.registry.write_schema_file_if_needed(entry, header)
        self.registry.record_drawing("P1-D1", header, entry, reused)
        self.registry.record_drawing("P1-D2", header, None, False, error="classification failed")

        manifest_path = self.registry.write_manifest()
        payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        self.assertEqual(len(payload["drawings"]), 2)
        by_id = {row["drawing_id"]: row for row in payload["drawings"]}
        self.assertEqual(by_id["P1-D1"]["schema_id"], entry.schema_id)
        self.assertEqual(by_id["P1-D2"]["error"], "classification failed")

    def test_output_dir_created_automatically(self):
        nested = Path(self.tmpdir.name) / "nested" / "schemas"
        registry = SchemaRegistry(str(nested), "20260101_120000")
        self.assertTrue(nested.exists())
        self.assertEqual(registry.output_dir, nested)


class TestInvalidSchemaRejected(unittest.TestCase):
    def test_check_schema_rejects_malformed_document(self):
        with self.assertRaises(SchemaValidationError):
            validate_schema({"type": "definitely-not-a-real-type"})


if __name__ == "__main__":
    unittest.main()
