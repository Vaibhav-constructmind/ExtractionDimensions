"""Environment configuration for the extraction pipeline.

All secrets/endpoints are read from environment variables (populated from a
local `.env` file via python-dotenv). Nothing here should ever hard-code a
real endpoint or key.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Load .env once, on import. Safe to call multiple times; does not override
# variables already set in the real environment.
load_dotenv()

REQUIRED_VARS = [
    "AZURE_DOC_INTEL_ENDPOINT",
    "AZURE_DOC_INTEL_KEY",
    "AZURE_FOUNDRY_ENDPOINT",
    "AZURE_FOUNDRY_API_KEY",
]


class ConfigError(Exception):
    """Raised when required configuration is missing or invalid."""


@dataclass(frozen=True)
class Settings:
    doc_intel_endpoint: str
    doc_intel_key: str
    foundry_endpoint: str
    foundry_api_key: str
    foundry_claude_deployment: str
    render_dpi: int
    detail_render_dpi: int


def missing_vars() -> list[str]:
    """Return the list of required env var names that are unset/blank."""
    return [name for name in REQUIRED_VARS if not os.environ.get(name, "").strip()]


def load_settings() -> Settings:
    """Load and validate settings, raising ConfigError listing any gaps."""
    gaps = missing_vars()
    if gaps:
        raise ConfigError(
            "Missing required configuration: " + ", ".join(gaps) +
            ". Fill these in your .env file (see .env.example)."
        )

    try:
        render_dpi = int(os.environ.get("RENDER_DPI", "200"))
    except ValueError as exc:
        raise ConfigError("RENDER_DPI must be an integer") from exc

    try:
        detail_render_dpi = int(os.environ.get("DETAIL_RENDER_DPI", "600"))
    except ValueError as exc:
        raise ConfigError("DETAIL_RENDER_DPI must be an integer") from exc

    return Settings(
        doc_intel_endpoint=os.environ["AZURE_DOC_INTEL_ENDPOINT"].strip(),
        doc_intel_key=os.environ["AZURE_DOC_INTEL_KEY"].strip(),
        foundry_endpoint=os.environ["AZURE_FOUNDRY_ENDPOINT"].strip(),
        foundry_api_key=os.environ["AZURE_FOUNDRY_API_KEY"].strip(),
        foundry_claude_deployment=os.environ.get(
            "AZURE_FOUNDRY_CLAUDE_DEPLOYMENT", "claude-sonnet-5"
        ).strip(),
        render_dpi=render_dpi,
        detail_render_dpi=detail_render_dpi,
    )
