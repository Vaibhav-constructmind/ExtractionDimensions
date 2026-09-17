"""Pydantic models for the extraction pipeline's output.

Each Drawing nests its own metadata, dimensions, and elevation datums --
everything found on that drawing lives under it, and the top-level
ExtractionResult is the single JSON covering every drawing in the PDF.
"""
from __future__ import annotations

# pyrefly: ignore [missing-import]
from pydantic import BaseModel, Field


class StepFormula(BaseModel):
    """Riser/tread breakdown backing a stair-rise dimension, e.g. '15 RISERS @ 175 = 2625'."""

    count: float | None = Field(None, description="Number of risers/treads, e.g. 15")
    riser_or_tread_dim: float | None = Field(None, description="Height/depth per riser or tread, e.g. 175")
    calculated_total: float | None = Field(None, description="count * riser_or_tread_dim, e.g. 2625")


class Dimension(BaseModel):
    dimension_id: str = Field(..., description="Unique ID, e.g. P1-D2-DIM03")
    orientation: str | None = Field(None, description="vertical | horizontal | diagonal | other")
    section_part: str | None = Field(
        None, description="Where on the drawing this is, e.g. 'Shaft / Vent Alcove (Upper-Left)'"
    )
    element: str | None = Field(
        None, description="What is being dimensioned, e.g. 'Ventilation intake shaft clear height'"
    )
    value: float | None = Field(None, description="Numeric value in `unit`, e.g. 5550. Omit if illegible.")
    unit: str | None = Field(None, description="e.g. mm, m, ft, in")
    start_reference: str | None = Field(None, description="Where the dimension/witness line starts, e.g. 'Top of Slab (+601.88 T.O.S.)'")
    end_reference: str | None = Field(None, description="Where the dimension/witness line ends, e.g. 'Underside of top concrete roof slab'")
    label_text: str = Field(
        ..., description="Raw text exactly as printed on the drawing, e.g. '5550' or '15 RISERS @ 175 = 2625'; 'unclear' if illegible"
    )
    type: str | None = Field(
        None,
        description=(
            "clearance | headroom | floor_to_floor | level_difference | guardrail_height | "
            "stair_rise | tread_going | wall_thickness | width | other"
        ),
    )
    step_formula: StepFormula | None = Field(
        None, description="Only for stair_rise dimensions expressed as a riser/tread count formula"
    )
    confidence: str | None = Field(None, description="high | medium | low -- low when the reading is uncertain")
    notes: str | None = Field(
        None, description="Context explaining an uncertain/illegible reading, or a math cross-check discrepancy"
    )
    page_number: int


class ElevationDatum(BaseModel):
    datum_id: str = Field(..., description="Unique ID, e.g. P1-D2-DATUM01")
    section_part: str | None = Field(None, description="Where this datum is called out, e.g. 'Top Exit Landing Threshold'")
    level_code: str | None = Field(None, description="e.g. FFL, TOS, TOC, SSL")
    elevation_value: float | None = Field(None, description="e.g. 605.452")
    unit: str | None = Field(None, description="e.g. m")
    label_text: str = Field(..., description="Raw text as printed, e.g. '605.452 F.F.L.'; 'unclear' if illegible")
    confidence: str | None = Field(None, description="high | medium | low -- low when the reading is uncertain")
    notes: str | None = Field(None, description="Context explaining an uncertain/illegible reading")
    page_number: int


class QuantityField(BaseModel):
    """One derived/extracted quantity-takeoff figure, with how it was obtained."""

    value: float | None = Field(None, description="Numeric value in `unit`. Omit/null if not derivable from this drawing.")
    unit: str | None = Field(None, description="e.g. mm, m, m2, m3")
    method: str | None = Field(
        None,
        description=(
            "How this was obtained, e.g. 'count of stair_rise dimensions', "
            "'sum of riser counts across flights', 'FFL(top) - FFL(bottom)', or "
            "'not derivable -- requires plan view showing wall centerlines/thickness'"
        ),
    )
    confidence: str | None = Field(None, description="high | medium | low -- low when derived/estimated rather than directly labeled")
    notes: str | None = Field(None, description="Caveats, ambiguity, or what additional drawing (e.g. plan view) would be needed")


class QuantityTakeoff(BaseModel):
    """Stair/enclosure quantity-takeoff figures derived from this drawing's dimensions and datums.

    Vertical/stair-geometry fields (flights, risers, treads, landings, vertical drop) are
    typically derivable from a section view's stair_rise dimensions and elevation_datums.
    Wall/footprint/concrete/formwork fields normally require a plan view (footprint,
    wall centerlines, thickness) and are left null with an explanatory note when the
    source drawing doesn't show them.
    """

    num_flights: QuantityField | None = None
    num_risers_total: QuantityField | None = None
    riser_height: QuantityField | None = None
    num_treads_total: QuantityField | None = None
    tread_length: QuantityField | None = Field(None, description="Tread going/depth, per tread")
    total_tread_length: QuantityField | None = Field(None, description="Sum of horizontal tread run across all flights")
    num_landings: QuantityField | None = None
    total_vertical_drop: QuantityField | None = Field(None, description="Overall vertical rise, top FFL to bottom FFL")
    perimeter_wall_length: QuantityField | None = None
    internal_room_footprint_area: QuantityField | None = None
    inner_perimeter: QuantityField | None = None
    wall_thickness: QuantityField | None = None
    centerline_perimeter: QuantityField | None = None
    perimeter_wall_concrete_quantity: QuantityField | None = None
    flight_landing_concrete_quantity: QuantityField | None = None
    wall_formwork_area: QuantityField | None = None
    soffit_stair_formwork_area: QuantityField | None = None


class BoundingBox(BaseModel):
    """A drawing's extent on its page, as fractions of the full page width/height
    (0,0 = top-left corner of the page, 1,1 = bottom-right). Used to crop and
    re-render this drawing alone at higher resolution in a later pass."""

    x0: float = Field(..., ge=0.0, le=1.0, description="Left edge, as a fraction of page width")
    y0: float = Field(..., ge=0.0, le=1.0, description="Top edge, as a fraction of page height")
    x1: float = Field(..., ge=0.0, le=1.0, description="Right edge, as a fraction of page width")
    y1: float = Field(..., ge=0.0, le=1.0, description="Bottom edge, as a fraction of page height")
    confidence: str | None = Field(None, description="high | medium | low -- low when the drawing's extent is ambiguous")
    notes: str | None = Field(None, description="Context, e.g. why the extent is uncertain or was padded")


class DrawingMetadata(BaseModel):
    project_name: str | None = Field(None, description="Project name from the title block, if visible")
    drawing_title: str | None = Field(None, description="Drawing title/callout, e.g. 'Stair Details / Stair 00801'")
    drawing_number: str | None = Field(None, description="Drawing/sheet number as printed on the title block, e.g. '2738-S13-H-26-S-512'")
    sheet_scale: str | None = Field(None, description="e.g. '1:50'")
    default_units: str | None = Field(None, description="Predominant unit used on this drawing, e.g. 'mm'")
    revision: str | None = Field(None, description="Revision code/status from the title block, if visible")


class Drawing(BaseModel):
    drawing_id: str = Field(..., description="Unique ID assigned by this pipeline, e.g. P1-D2")
    page_number: int
    drawing_type: str = Field(
        ..., description="elevation | section | isometric | plan | detail | schedule | other"
    )
    drawing_metadata: DrawingMetadata = Field(default_factory=DrawingMetadata)
    dimensions: list[Dimension] = Field(default_factory=list)
    elevation_datums: list[ElevationDatum] = Field(default_factory=list)
    quantity_takeoff: QuantityTakeoff | None = Field(
        default=None, description="Stair/enclosure quantities derived from this drawing's dimensions and datums"
    )
    bounding_box: BoundingBox | None = Field(
        default=None,
        description="This drawing's extent on the page (fractions of page width/height), for a later crop-and-re-render pass",
    )
    detail_pass_applied: bool = Field(
        default=False,
        description=(
            "True if dimensions/elevation_datums/quantity_takeoff came from a second pass that "
            "cropped this drawing's bounding_box out of the page and re-rendered just that region "
            "at a much higher effective DPI, rather than from the single full-page render."
        ),
    )


class ExtractionResult(BaseModel):
    source_file: str
    total_pages: int
    total_drawings: int
    drawings: list[Drawing] = Field(default_factory=list)
