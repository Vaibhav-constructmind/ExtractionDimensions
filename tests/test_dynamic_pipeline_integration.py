"""Integration test for the new dynamic-schema pipeline stage wired into
run_pipeline: classification -> schema registry get-or-create -> schema
file written to disk -> dynamic_takeoff computed -> manifest + result
fields populated. Every external boundary (OCR, rendering, the LLM) is
mocked, so this exercises only pipeline/orchestrator.py's own wiring.

Run with:
    .venv\\Scripts\\python.exe -m unittest tests.test_dynamic_pipeline_integration -v
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.config import Settings
from pipeline.orchestrator import run_pipeline

_PAGE_IMAGE = b"fake-page-image"
_CROP_IMAGE = b"fake-crop-image"

_RAW_DRAWING = {
    "drawing_type": "plan",
    "drawing_metadata": {"drawing_title": "FOOTING PLAN", "title_callout_number": "1"},
    "dimensions": [
        {"dimension_id": "ignored", "type": "other", "value": 5000.0, "unit": "mm",
         "element": "footing length", "label_text": "5000"},
        {"dimension_id": "ignored", "type": "other", "value": 2000.0, "unit": "mm",
         "element": "footing width", "label_text": "2000"},
        {"dimension_id": "ignored", "type": "other", "value": 600.0, "unit": "mm",
         "element": "footing depth", "label_text": "600"},
    ],
    "elevation_datums": [],
    "detail_callouts": [],
    "quantity_takeoff": None,
    "bounding_box": {"x0": 0.0, "y0": 0.0, "x1": 1.0, "y1": 1.0, "confidence": "high"},
}

_CLASSIFICATION_RESULT = {
    "classification": {
        "project_type": "commercial building",
        "discipline": "structural",
        "drawing_type": "plan",
        "building_elements": ["foundation", "footing"],
        "confidence": "high",
    },
    "quantity_items": [
        {
            "name": "footing_concrete_volume",
            "description": "Footing concrete volume",
            "unit": "m3",
            "measurement_basis": "volume",
            "resource_category": "material",
            "depends_on": ["footing_length", "footing_width", "footing_depth"],
            "formula": "footing_length * footing_width * footing_depth",
        }
    ],
}


def _fake_extract_page(image_bytes, ocr_text, page_number, settings, dynamic_properties=None):
    # Pass 1 (whole page, no dynamic_properties) returns the plain drawing.
    # The detail pass (per drawing, dynamic_properties merged in once
    # classification has proposed this drawing's input roles) additionally
    # reports a value for each requested role directly, simulating the
    # model reading them off the drawing the same way real extraction would.
    drawing = dict(_RAW_DRAWING)
    if dynamic_properties:
        drawing["quantities"] = {role: 5000.0 if role == "footing_length" else
                                  2000.0 if role == "footing_width" else 600.0
                                  for role in dynamic_properties}
    return [drawing]


def _fake_classify_drawing(image_bytes, ocr_text, page_number, settings):
    return _CLASSIFICATION_RESULT


class TestDynamicSchemaPipelineIntegration(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.settings = Settings(
            doc_intel_endpoint="https://example.invalid",
            doc_intel_key="key",
            foundry_endpoint="https://example.invalid",
            foundry_api_key="key",
            foundry_claude_deployment="claude-sonnet-5",
            render_dpi=200,
            detail_render_dpi=600,
            schema_output_dir=self.tmpdir.name,
        )

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_generic_drawing_gets_schema_file_and_dynamic_takeoff(self):
        with (
            patch("pipeline.orchestrator.doc_intelligence.analyze_pdf", return_value=object()),
            patch("pipeline.orchestrator.doc_intelligence.text_by_page", return_value={1: ""}),
            patch("pipeline.orchestrator.doc_intelligence.lines_by_page", return_value={1: []}),
            patch("pipeline.orchestrator.doc_intelligence.page_count", return_value=1),
            patch("pipeline.orchestrator.render.render_pages", return_value=[_PAGE_IMAGE]),
            patch("pipeline.orchestrator.render.render_crop", return_value=_CROP_IMAGE),
            patch("pipeline.orchestrator.claude_extractor.extract_page", side_effect=_fake_extract_page),
            patch("pipeline.orchestrator.claude_extractor.classify_drawing", side_effect=_fake_classify_drawing),
        ):
            result = run_pipeline(b"fake-pdf-bytes", "test.pdf", self.settings)

        self.assertEqual(len(result.drawings), 1)
        drawing = result.drawings[0]

        self.assertIsNotNone(drawing.classification)
        self.assertEqual(drawing.classification.discipline, "structural")

        self.assertIsNotNone(drawing.dynamic_takeoff)
        quantities = drawing.dynamic_takeoff.quantities
        self.assertEqual(len(quantities), 1)
        q = quantities[0]
        self.assertIsNotNone(q.value)
        self.assertAlmostEqual(q.value.value, 5000.0 * 2000.0 * 600.0)

        # Schema file + manifest actually landed on disk in SCHEMA_OUTPUT_DIR.
        self.assertEqual(result.schema_output_dir, self.tmpdir.name)
        self.assertEqual(len(result.schema_files_written), 1)
        self.assertTrue(Path(result.schema_files_written[0]).exists())
        self.assertTrue(Path(result.schema_manifest_path).exists())

        manifest = json.loads(Path(result.schema_manifest_path).read_text(encoding="utf-8"))
        self.assertEqual(len(manifest["drawings"]), 1)
        self.assertEqual(manifest["drawings"][0]["schema_id"], drawing.dynamic_takeoff.schema_id)

    def test_second_run_adds_new_timestamped_files_without_overwriting(self):
        # The registry is scoped to one run (per spec section 5: "Maintain a
        # schema registry for the run") -- reuse/extend applies to several
        # drawings sharing a classification WITHIN one run, not across
        # separate pipeline invocations. Two runs of the same PDF therefore
        # each get their own schema file and manifest, both landing in the
        # same folder without clobbering each other.
        with (
            patch("pipeline.orchestrator.doc_intelligence.analyze_pdf", return_value=object()),
            patch("pipeline.orchestrator.doc_intelligence.text_by_page", return_value={1: ""}),
            patch("pipeline.orchestrator.doc_intelligence.lines_by_page", return_value={1: []}),
            patch("pipeline.orchestrator.doc_intelligence.page_count", return_value=1),
            patch("pipeline.orchestrator.render.render_pages", return_value=[_PAGE_IMAGE]),
            patch("pipeline.orchestrator.render.render_crop", return_value=_CROP_IMAGE),
            patch("pipeline.orchestrator.claude_extractor.extract_page", side_effect=_fake_extract_page),
            patch("pipeline.orchestrator.claude_extractor.classify_drawing", side_effect=_fake_classify_drawing),
        ):
            result1 = run_pipeline(b"fake-pdf-bytes", "test.pdf", self.settings)
            result2 = run_pipeline(b"fake-pdf-bytes", "test.pdf", self.settings)

        self.assertNotEqual(result1.schema_manifest_path, result2.schema_manifest_path)
        self.assertEqual(len(result1.schema_files_written), 1)
        self.assertEqual(len(result2.schema_files_written), 1)
        self.assertNotEqual(result1.schema_files_written[0], result2.schema_files_written[0])
        self.assertTrue(Path(result1.schema_files_written[0]).exists())
        self.assertTrue(Path(result2.schema_files_written[0]).exists())

        all_files = list(Path(self.tmpdir.name).glob("*.json"))
        # 2 schema files + 2 manifests, one pair per run, nothing overwritten.
        self.assertEqual(len(all_files), 4)

    def test_directly_reported_quantities_take_priority_over_dimension_matching(self):
        # The detail pass reports different numbers than what the plain
        # dimensions would heuristically match to -- the reported ones must
        # win, proving the merged-schema values flow through end to end.
        def fake_extract_page_with_override(image_bytes, ocr_text, page_number, settings, dynamic_properties=None):
            drawing = dict(_RAW_DRAWING)
            if dynamic_properties:
                drawing["quantities"] = {"footing_length": 10.0, "footing_width": 20.0, "footing_depth": 30.0}
            return [drawing]

        with (
            patch("pipeline.orchestrator.doc_intelligence.analyze_pdf", return_value=object()),
            patch("pipeline.orchestrator.doc_intelligence.text_by_page", return_value={1: ""}),
            patch("pipeline.orchestrator.doc_intelligence.lines_by_page", return_value={1: []}),
            patch("pipeline.orchestrator.doc_intelligence.page_count", return_value=1),
            patch("pipeline.orchestrator.render.render_pages", return_value=[_PAGE_IMAGE]),
            patch("pipeline.orchestrator.render.render_crop", return_value=_CROP_IMAGE),
            patch("pipeline.orchestrator.claude_extractor.extract_page", side_effect=fake_extract_page_with_override),
            patch("pipeline.orchestrator.claude_extractor.classify_drawing", side_effect=_fake_classify_drawing),
        ):
            result = run_pipeline(b"fake-pdf-bytes", "test.pdf", self.settings)

        q = result.drawings[0].dynamic_takeoff.quantities[0]
        self.assertIsNotNone(q.value)
        self.assertAlmostEqual(q.value.value, 10.0 * 20.0 * 30.0)
        self.assertTrue(any("reported" in s for s in q.sources))


if __name__ == "__main__":
    unittest.main()
