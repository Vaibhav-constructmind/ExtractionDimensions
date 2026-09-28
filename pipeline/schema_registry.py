"""Registry of generated quantity-takeoff JSON Schemas for one pipeline run.

Every drawing that goes through the generic (non-stair) dynamic-takeoff path
gets a classification (discipline + drawing_type) and a proposed list of
quantity items (see DynamicQuantityItemSpec in schema.py). This module:

  1. Reuses or extends an existing schema when a later drawing shares the
     same (discipline, drawing_type) key, so quantities from similar
     drawings across different files stay comparable/aggregatable, instead
     of minting a fresh schema per drawing.
  2. Turns a validated item list into an actual JSON Schema (Draft 2020-12)
     document and checks it against the meta-schema.
  3. Persists every schema as its own timestamped .json file in one fixed
     folder (SCHEMA_OUTPUT_DIR), plus one manifest per run mapping every
     drawing to the schema file it used.

Nothing here talks to the model -- this is pure bookkeeping/validation/I-O
over whatever DynamicQuantityItemSpec list the extractor proposed.
"""
from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import jsonschema

from .schema import DrawingClassification, DynamicQuantityItemSpec, SchemaHeader

logger = logging.getLogger(__name__)

_JSON_SCHEMA_DIALECT = "https://json-schema.org/draft/2020-12/schema"


class SchemaValidationError(ValueError):
    """Raised when a proposed/assembled takeoff schema fails meta-schema
    validation -- callers should retry schema generation with this error
    fed back, the same pattern claude_extractor.py already uses for
    malformed tool calls."""


def _normalize_key_part(value: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (value or "unknown").strip().lower()).strip("_") or "unknown"


def _sanitize_filename_part(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "unknown"


@dataclass
class SchemaRegistryEntry:
    schema_id: str
    version: int
    discipline_key: str
    drawing_type_key: str
    items: list[DynamicQuantityItemSpec] = field(default_factory=list)
    json_schema: dict = field(default_factory=dict)
    file_path: str | None = None  # path this version was last written to, if any


def build_json_schema(schema_id: str, version: int, items: list[DynamicQuantityItemSpec]) -> dict:
    """Assemble a Draft 2020-12 JSON Schema document for a `quantities`
    object whose properties are the given items -- each property carries
    its unit/measurement_basis/resource_category/depends_on/formula as
    custom (`x-*`) keywords, which is valid, extensible JSON Schema."""
    properties = {}
    for spec in items:
        properties[spec.name] = {
            "type": ["number", "null"],
            "description": spec.description or spec.name,
            "x-unit": spec.unit,
            "x-measurement-basis": spec.measurement_basis,
            "x-resource-category": spec.resource_category,
            "x-depends-on": list(spec.depends_on),
            "x-formula": spec.formula,
        }

    return {
        "$schema": _JSON_SCHEMA_DIALECT,
        "$id": f"urn:takeoff-schema:{schema_id}:v{version}",
        "title": f"Quantity takeoff schema {schema_id} v{version}",
        "type": "object",
        "properties": {"quantities": {"type": "object", "properties": properties}},
        "required": ["quantities"],
    }


def validate_schema(json_schema: dict) -> None:
    """Raise SchemaValidationError if `json_schema` doesn't itself validate
    against the Draft 2020-12 meta-schema."""
    try:
        jsonschema.Draft202012Validator.check_schema(json_schema)
    except jsonschema.exceptions.SchemaError as exc:
        raise SchemaValidationError(str(exc)) from exc


class SchemaRegistry:
    """In-memory registry for one pipeline run, keyed by
    (discipline, drawing_type). Also accumulates this run's manifest and
    writes both schema files and the manifest to SCHEMA_OUTPUT_DIR."""

    def __init__(self, output_dir: str, run_timestamp: str):
        self.output_dir = Path(output_dir)
        self.run_timestamp = run_timestamp
        self._entries: dict[tuple[str, str], SchemaRegistryEntry] = {}
        self._manifest_entries: list[dict] = []
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_or_create(
        self,
        classification: DrawingClassification,
        proposed_items: list[DynamicQuantityItemSpec],
    ) -> tuple[SchemaRegistryEntry, bool]:
        """Return (entry, reused) for this classification. `reused` is True
        when an existing entry for this (discipline, drawing_type) key was
        found -- its schema_id is kept, and any proposed item NOT already
        present (by name) is unioned in, bumping the version. A brand new
        key creates version 1.
        """
        key = (_normalize_key_part(classification.discipline), _normalize_key_part(classification.drawing_type))
        existing = self._entries.get(key)

        if existing is None:
            schema_id = uuid.uuid4().hex[:12]
            json_schema = build_json_schema(schema_id, 1, proposed_items)
            validate_schema(json_schema)
            entry = SchemaRegistryEntry(
                schema_id=schema_id,
                version=1,
                discipline_key=key[0],
                drawing_type_key=key[1],
                items=list(proposed_items),
                json_schema=json_schema,
            )
            self._entries[key] = entry
            return entry, False

        existing_names = {item.name for item in existing.items}
        new_items = [item for item in proposed_items if item.name not in existing_names]
        if new_items:
            existing.items = existing.items + new_items
            existing.version += 1
            existing.json_schema = build_json_schema(existing.schema_id, existing.version, existing.items)
            validate_schema(existing.json_schema)
            existing.file_path = None  # needs re-writing under the new version
        return existing, True

    def write_schema_file_if_needed(self, entry: SchemaRegistryEntry, header: SchemaHeader) -> str | None:
        """Write `entry`'s current version to disk if it hasn't been
        written yet (or was just extended to a new version). Returns the
        path written, or None if this exact version was already on disk
        (a genuinely reused, unchanged schema -- recorded in the manifest
        instead of being written again, per spec)."""
        if entry.file_path is not None:
            return None

        filename = (
            f"{self.run_timestamp}_{_sanitize_filename_part(header.source_pdf)}"
            f"_p{header.page}_d{header.drawing_index}_{entry.drawing_type_key}.schema.json"
        )
        path = self.output_dir / filename
        payload = {
            "header": header.model_dump(mode="json"),
            "schema_id": entry.schema_id,
            "version": entry.version,
            "json_schema": entry.json_schema,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        entry.file_path = str(path)
        logger.info("Wrote takeoff schema %s v%s to %s", entry.schema_id, entry.version, path)
        return str(path)

    def record_drawing(
        self,
        drawing_id: str,
        header: SchemaHeader,
        entry: SchemaRegistryEntry | None,
        reused: bool,
        error: str | None = None,
    ) -> None:
        """Add one row to this run's manifest -- called for every drawing
        that went through the dynamic-schema path, success or failure."""
        self._manifest_entries.append({
            "drawing_id": drawing_id,
            "source_pdf": header.source_pdf,
            "page": header.page,
            "drawing_index": header.drawing_index,
            "drawing_title": header.drawing_title,
            "classification": header.classification.model_dump(mode="json"),
            "schema_id": entry.schema_id if entry else None,
            "schema_version": entry.version if entry else None,
            "schema_file": entry.file_path if entry else None,
            "reused_existing_schema": reused,
            "error": error,
        })

    def write_manifest(self) -> str:
        path = self.output_dir / f"{self.run_timestamp}_manifest.json"
        payload = {
            "run_timestamp": self.run_timestamp,
            "schema_output_dir": str(self.output_dir),
            "drawings": self._manifest_entries,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("Wrote run manifest to %s", path)
        return str(path)

    @property
    def manifest_entries(self) -> list[dict]:
        return list(self._manifest_entries)

    @property
    def written_files(self) -> list[str]:
        return [e.file_path for e in self._entries.values() if e.file_path]
