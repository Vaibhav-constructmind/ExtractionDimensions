"""Azure Document Intelligence wrapper.

Runs the prebuilt-layout model over the uploaded PDF to get reliable OCR text
per page. This text is passed to Claude alongside the rendered page image so
small/rotated dimension strings (e.g. 3'-6") are read correctly even when the
image alone is ambiguous.
"""
from __future__ import annotations

from dataclasses import dataclass

# pyrefly: ignore [missing-import]
from azure.ai.documentintelligence import DocumentIntelligenceClient
# pyrefly: ignore [missing-import]
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest, AnalyzeResult
# pyrefly: ignore [missing-import]
from azure.core.credentials import AzureKeyCredential

from .config import Settings


@dataclass(frozen=True)
class OcrLine:
    """One recognized text line, positioned as fractions (0.0-1.0) of the
    full page's width/height -- the same convention as `BoundingBox`, so a
    line's position can be tested against a drawing's crop directly."""

    text: str
    x0: float
    y0: float
    x1: float
    y1: float


def analyze_pdf(pdf_bytes: bytes, settings: Settings) -> AnalyzeResult:
    """Run prebuilt-layout analysis over the PDF and return the raw result."""
    client = DocumentIntelligenceClient(
        endpoint=settings.doc_intel_endpoint,
        credential=AzureKeyCredential(settings.doc_intel_key),
    )
    poller = client.begin_analyze_document(
        "prebuilt-layout",
        AnalyzeDocumentRequest(bytes_source=pdf_bytes),
    )
    return poller.result()


def text_by_page(result: AnalyzeResult) -> dict[int, str]:
    """Join each page's recognized lines into a plain-text block, keyed by
    1-indexed page number."""
    pages_text: dict[int, str] = {}
    for page in result.pages or []:
        page_number = page.page_number
        lines = [line.content for line in (page.lines or [])]
        pages_text[page_number] = "\n".join(lines)
    return pages_text


def lines_by_page(result: AnalyzeResult) -> dict[int, list[OcrLine]]:
    """Like `text_by_page`, but keeping each line's position as fractions of
    the page's width/height, so a caller can scope OCR text down to just the
    lines that fall within one drawing's cropped region instead of handing
    every drawing the whole page's OCR text undifferentiated."""
    pages_lines: dict[int, list[OcrLine]] = {}
    for page in result.pages or []:
        page_number = page.page_number
        width, height = page.width, page.height
        lines: list[OcrLine] = []
        for line in page.lines or []:
            polygon = line.polygon or []
            if not width or not height or len(polygon) < 8:
                continue
            xs = polygon[0::2]
            ys = polygon[1::2]
            lines.append(
                OcrLine(
                    text=line.content,
                    x0=min(xs) / width,
                    y0=min(ys) / height,
                    x1=max(xs) / width,
                    y1=max(ys) / height,
                )
            )
        pages_lines[page_number] = lines
    return pages_lines


def page_count(result: AnalyzeResult) -> int:
    return len(result.pages or [])
