"""Render PDF pages to images for vision input, using PyMuPDF."""
from __future__ import annotations

import pymupdf as fitz  # PyMuPDF (the `fitz` import name is deprecated)

# Claude's vision endpoint hard-rejects any image over 8000px on a side, and
# internally resizes anything bigger than ~1568px on the long edge / ~1.15
# effective megapixels before it even looks at it -- so rendering large
# architectural sheets (e.g. ARCH D/E, 24x36in+) at a high DPI both risks a
# 400 (dimensions exceed max allowed size) and wastes upload bandwidth on
# detail Claude will never see. We cap the render scale per page to whichever
# limit is most restrictive for that page's size.
MAX_LONG_EDGE_PX = 1568
MAX_MEGAPIXELS = 1_150_000
HARD_MAX_DIM_PX = 8000  # API's actual rejection threshold


def _capped_zoom(width_pt: float, height_pt: float, zoom: float) -> float:
    width_px = width_pt * zoom
    height_px = height_pt * zoom
    long_edge = max(width_px, height_px)
    if long_edge <= 0:
        return zoom

    scale = min(
        1.0,
        MAX_LONG_EDGE_PX / long_edge,
        HARD_MAX_DIM_PX / long_edge,
        (MAX_MEGAPIXELS / (width_px * height_px)) ** 0.5,
    )
    return zoom * scale


def render_pages(pdf_bytes: bytes, dpi: int = 200) -> list[bytes]:
    """Rasterize each page of the PDF to PNG bytes, capped to stay within
    Claude's vision size limits.

    Returns a list ordered by page (index 0 == page 1).
    """
    images: list[bytes] = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page in doc:
            base_zoom = dpi / 72.0  # PDF points are 72/inch
            zoom = _capped_zoom(page.rect.width, page.rect.height, base_zoom)
            pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
            images.append(pixmap.tobytes("png"))
    return images


def render_crop(
    pdf_bytes: bytes,
    page_number: int,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    max_dpi: int = 600,
) -> bytes:
    """Rasterize just one region of one page (e.g. a single drawing's bounding
    box) to PNG bytes, at the highest resolution that still fits Claude's
    vision size limits.

    x0/y0/x1/y1 are fractions (0.0-1.0) of the full page's width/height, with
    (0,0) at the top-left corner -- the same convention as `BoundingBox`.
    `max_dpi` is a ceiling, not a target: because the crop covers far less
    physical area than the full page, `_capped_zoom` usually has room to land
    much closer to (or at) `max_dpi` than a full-page render ever could,
    without needing per-page tuning here.
    """
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        page = doc[page_number - 1]
        full_rect = page.rect
        crop_rect = fitz.Rect(
            full_rect.x0 + x0 * full_rect.width,
            full_rect.y0 + y0 * full_rect.height,
            full_rect.x0 + x1 * full_rect.width,
            full_rect.y0 + y1 * full_rect.height,
        )

        base_zoom = max_dpi / 72.0
        zoom = _capped_zoom(crop_rect.width, crop_rect.height, base_zoom)
        pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=crop_rect)
        return pixmap.tobytes("png")
