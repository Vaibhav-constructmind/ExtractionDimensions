# Drawing Dimension Extraction

Upload a PDF of architectural/structural drawings and get back structured JSON:
how many individual drawings are on the sheet(s), what type each one is
(elevation, section, isometric, plan, detail, ...), and every dimension found
on each drawing — each with a unique ID so it can be referenced back to its
drawing.

Pipeline: **Azure Document Intelligence** (`prebuilt-layout`) extracts reliable
OCR text per page; **Claude Sonnet-5**, called through your **Azure AI Foundry**
deployment, looks at the rendered page image plus that OCR text to segment the
page into distinct drawings, classify each one, and extract its dimensions.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
copy .env.example .env
```

Fill in `.env` with:
- `AZURE_DOC_INTEL_ENDPOINT` / `AZURE_DOC_INTEL_KEY` — your Azure Document Intelligence resource.
- `AZURE_FOUNDRY_ENDPOINT` / `AZURE_FOUNDRY_API_KEY` — your Azure AI Foundry project's
  Anthropic-compatible endpoint + key for the Claude Sonnet-5 deployment.
- `AZURE_FOUNDRY_CLAUDE_DEPLOYMENT` — the deployment/model name (defaults to `claude-sonnet-5`).

> **Note on the Foundry endpoint:** this project calls Claude through the official
> `anthropic` Python SDK with `base_url` pointed at your Foundry endpoint, since Foundry's
> Claude deployments expose an Anthropic Messages-compatible API. If your specific
> deployment's wire format differs, the only place that needs adjusting is
> `pipeline/claude_extractor.py` (`_build_client` / `_call_once`).

## Run

```bash
streamlit run app.py
```

Open the URL Streamlit prints, upload a PDF, and click **Run extraction**. The sidebar
shows which required configuration values are still missing.

## Output shape

```json
{
  "source_file": "drawing.pdf",
  "total_pages": 2,
  "total_drawings": 3,
  "drawings": [
    {
      "drawing_id": "P1-D1",
      "page_number": 1,
      "drawing_type": "section",
      "title": "STAIR SECTION A-A",
      "dimensions": [
        {
          "id": "P1-D1-DIM01",
          "value": "3'-6\"",
          "normalized": "3.5 ft",
          "label": "riser height",
          "page_number": 1
        }
      ]
    }
  ]
}
```

`drawing_id` and dimension `id` are stable unique tags (page + drawing index +
dimension index) used to reference a value back to its drawing; no pixel
bounding boxes are produced by design.

## Project layout

- `app.py` — Streamlit UI.
- `pipeline/config.py` — env var loading/validation.
- `pipeline/schema.py` — pydantic models for the output (`Dimension`, `Drawing`, `ExtractionResult`).
- `pipeline/doc_intelligence.py` — Azure Document Intelligence wrapper (OCR/layout).
- `pipeline/render.py` — renders PDF pages to images (PyMuPDF) for vision input.
- `pipeline/claude_extractor.py` — Claude tool-use call that segments/classifies/extracts per page.
- `pipeline/orchestrator.py` — runs the full pipeline and assembles the final result.
