"""Unit tests for pipeline/schema_registry.py: JSON Schema assembly + meta-
schema validation, per-drawing independent schema generation (no reuse or
extension across drawings, even ones sharing a classification), and
file/manifest I/O.

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


class TestSchemaRegistryPerDrawingIndependence(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.registry = SchemaRegistry(self.tmpdir.name, "20260101_120000")

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_new_drawing_creates_version_1(self):
        entry = self.registry.create_for_drawing(_classification(), [_item()])
        self.assertEqual(entry.version, 1)

    def test_same_classification_still_gets_a_distinct_schema_id(self):
        # Two drawings sharing a classification must NOT share a schema --
        # each drawing's schema is independent, even if their proposed
        # items happen to be identical.
        entry1 = self.registry.create_for_drawing(_classification(), [_item()])
        entry2 = self.registry.create_for_drawing(_classification(), [_item()])
        self.assertNotEqual(entry1.schema_id, entry2.schema_id)

    def test_first_drawings_items_are_unaffected_by_a_later_drawings_items(self):
        entry1 = self.registry.create_for_drawing(_classification(), [_item()])
        entry2 = self.registry.create_for_drawing(
            _classification(), [_item(), _item(name="rebar_weight", depends_on=["length"], formula="length * 7.85")]
        )
        self.assertEqual({i.name for i in entry1.items}, {"footing_concrete_volume"})
        self.assertEqual({i.name for i in entry2.items}, {"footing_concrete_volume", "rebar_weight"})
        self.assertNotEqual(entry1.schema_id, entry2.schema_id)

    def test_different_classification_gets_different_schema_id(self):
        entry1 = self.registry.create_for_drawing(_classification(discipline="structural"), [_item()])
        entry2 = self.registry.create_for_drawing(_classification(discipline="MEP"), [_item()])
        self.assertNotEqual(entry1.schema_id, entry2.schema_id)


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
        entry = self.registry.create_for_drawing(classification, [_item()])
        header = self._header(classification).model_copy(update={"schema_id": entry.schema_id, "version": entry.version})
        path = self.registry.write_schema_file_if_needed(entry, header)
        self.assertIsNotNone(path)
        self.assertTrue(Path(path).exists())
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        self.assertEqual(payload["header"]["source_pdf"], "test.pdf")
        self.assertEqual(payload["schema_id"], entry.schema_id)

    def test_write_schema_file_is_noop_when_already_written(self):
        classification = _classification()
        entry = self.registry.create_for_drawing(classification, [_item()])
        header = self._header(classification).model_copy(update={"schema_id": entry.schema_id, "version": entry.version})
        first = self.registry.write_schema_file_if_needed(entry, header)
        second = self.registry.write_schema_file_if_needed(entry, header)
        self.assertIsNotNone(first)
        self.assertIsNone(second)

    def test_two_drawings_each_get_their_own_written_file(self):
        classification = _classification()
        entry1 = self.registry.create_for_drawing(classification, [_item()])
        header1 = self._header(classification).model_copy(update={"schema_id": entry1.schema_id, "version": entry1.version})
        path1 = self.registry.write_schema_file_if_needed(entry1, header1)

        entry2 = self.registry.create_for_drawing(
            classification, [_item(name="new_item", depends_on=["length"], formula="length")]
        )
        header2 = self._header(classification).model_copy(
            update={"schema_id": entry2.schema_id, "version": entry2.version, "drawing_index": 2}
        )
        path2 = self.registry.write_schema_file_if_needed(entry2, header2)

        self.assertIsNotNone(path1)
        self.assertIsNotNone(path2)
        self.assertNotEqual(path1, path2)

    def test_manifest_records_every_drawing(self):
        classification = _classification()
        entry = self.registry.create_for_drawing(classification, [_item()])
        header = self._header(classification).model_copy(update={"schema_id": entry.schema_id, "version": entry.version})
        self.registry.write_schema_file_if_needed(entry, header)
        self.registry.record_drawing("P1-D1", header, entry)
        self.registry.record_drawing("P1-D2", header, None, error="classification failed")

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
