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


class MeasuredValue(BaseModel):
    """A single numeric value with its unit -- used as an auditable input or
    output of a deterministic (non-LLM) calculation."""

    value: float
    unit: str
    source: str | None = Field(
        None, description="dimension_id this value came from, if traceable -- never fabricated"
    )


class DrawingClassification(BaseModel):
    """What kind of drawing this is, per the model's own judgement -- drives
    which quantity-takeoff schema gets generated/reused."""

    project_type: str | None = Field(None, description="e.g. 'commercial building', 'road/highway', 'tunnel'")
    discipline: str | None = Field(None, description="e.g. 'architectural', 'structural', 'MEP', 'civil'")
    drawing_type: str | None = Field(None, description="plan | section | elevation | detail | schedule | isometric")
    building_elements: list[str] = Field(
        default_factory=list,
        description="Building elements/systems shown, e.g. ['stair', 'handrail'] or ['foundation', 'footing']",
    )
    confidence: str | None = Field(None, description="high | medium | low -- low when the drawing's subject is ambiguous")
    notes: str | None = Field(None, description="Context for an uncertain classification")


class DynamicQuantityItemSpec(BaseModel):
    """One quantity-takeoff item proposed by the model for a given drawing
    classification -- the metadata needed to compute it deterministically in
    Python (pipeline.calculations), never by the model itself.

    `depends_on` is a list of logical input roles (e.g. 'wall_length',
    'wall_height'), not literal dimension_ids -- the same spec is reused
    across every drawing that shares this classification, and each drawing's
    own dimensions are matched to these roles at compute time (see
    pipeline.handlers._match_dimension_for_role). A role with no matching
    dimension on a given drawing simply produces a null quantity with a
    reason, never a guess.
    """

    name: str = Field(..., description="Quantity item name, e.g. 'wall_concrete_volume'")
    description: str | None = None
    unit: str = Field(..., description="e.g. m3, m2, m, count, kg")
    measurement_basis: str = Field(..., description="count | length | area | volume | weight")
    resource_category: str = Field(..., description="material | labor | equipment")
    depends_on: list[str] = Field(default_factory=list, description="Logical input roles this formula needs")
    formula: str = Field(..., description="Restricted arithmetic expression over `depends_on` names, e.g. 'wall_length * wall_height * wall_thickness'")


class SchemaHeader(BaseModel):
    """Traceability header stored inside a saved schema file and echoed back
    onto every drawing that used it."""

    schema_id: str
    version: int
    run_timestamp: str
    source_pdf: str
    page: int
    drawing_index: int
    drawing_title: str | None = None
    classification: DrawingClassification


class ComputedQuantity(BaseModel):
    """One quantity computed deterministically in Python from a
    DynamicQuantityItemSpec's formula: the LLM proposes the formula/inputs,
    Python evaluates it."""

    name: str
    unit: str
    measurement_basis: str
    resource_category: str
    formula: str
    value: MeasuredValue | None = Field(
        None, description="Null when one or more required inputs couldn't be matched on this drawing"
    )
    sources: list[str] = Field(default_factory=list, description="dimension_ids the inputs were matched to")
    reason: str | None = Field(None, description="Why value is null, when it is -- never a fabricated fallback")


class DynamicTakeoff(BaseModel):
    """Generic, non-stair quantity takeoff for one drawing: which schema was
    used (for cross-file comparability) and the computed quantities."""

    schema_id: str
    schema_version: int
    quantities: list[ComputedQuantity] = Field(default_factory=list)


class DetailCallout(BaseModel):
    """A numbered detail-bubble cross-reference on a drawing, e.g. a circled
    '6' pointing to 'STEEL HANDRAIL DETAIL-1-5' on another sheet. These are
    real annotated content but aren't a dimension or elevation datum, so they
    need their own record rather than being dropped."""

    callout_id: str = Field(..., description="Unique ID, e.g. P1-D2-CALLOUT01")
    callout_number: str | None = Field(None, description="The number/tag inside the circle/bubble, e.g. '6'")
    title: str | None = Field(None, description="The callout's label text, e.g. 'STEEL HANDRAIL DETAIL-1-5'")
    target_drawing_number: str | None = Field(
        None, description="The referenced sheet/drawing number printed under the callout, if shown, e.g. 'XXX-DWG-AR-AR-510002'"
    )
    section_part: str | None = Field(None, description="Where on the drawing this callout is, e.g. 'Top of upper flight, near grid 4'")
    label_text: str = Field(..., description="Raw text as printed, e.g. '6 STEEL HANDRAIL DETAIL-1-5'; 'unclear' if illegible")
    confidence: str | None = Field(None, description="high | medium | low -- low when the reading is uncertain")
    notes: str | None = Field(None, description="Context explaining an uncertain/illegible reading")
    page_number: int


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
    drawing_title: str | None = Field(
        None,
        description=(
            "This drawing's own title callout text (e.g. 'STAIR-07-INTERMEDIATE LANDING-02'), read "
            "from its title_callout_number circle -- never an internal equipment/room tag that "
            "happens to appear inside the drawing's geometry."
        ),
    )
    title_callout_number: str | None = Field(
        None,
        description=(
            "The number inside this drawing's own title-callout circle/bubble (usually bottom-left "
            "of its frame), e.g. '4'. On a sheet of N drawings these are typically sequential 1..N, "
            "one per drawing -- used to cross-check that drawing_title was read from the right place "
            "and wasn't duplicated from/confused with a neighboring drawing."
        ),
    )
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
    detail_callouts: list[DetailCallout] = Field(
        default_factory=list, description="Numbered detail-bubble cross-references found on this drawing"
    )
    classification: DrawingClassification | None = Field(
        default=None, description="What kind of drawing this is, per the model's classification pass"
    )
    dynamic_takeoff: DynamicTakeoff | None = Field(
        default=None,
        description=(
            "Resource-planning takeoff for this drawing -- schema/formulas proposed by the model "
            "per its classification, values computed deterministically by pipeline.calculations. "
            "None when classification/schema generation failed or proposed no quantity items."
        ),
    )
    bounding_box: BoundingBox | None = Field(
        default=None,
        description="This drawing's extent on the page (fractions of page width/height), for a later crop-and-re-render pass",
    )
    detail_pass_applied: bool = Field(
        default=False,
        description=(
            "True if dimensions/elevation_datums came from a second pass that cropped this "
            "drawing's bounding_box out of the page and re-rendered just that region at a much "
            "higher effective DPI, rather than from the single full-page render."
        ),
    )


class ExtractionResult(BaseModel):
    source_file: str
    total_pages: int
    total_drawings: int
    drawings: list[Drawing] = Field(default_factory=list)
    schema_output_dir: str | None = Field(
        None, description="Folder this run's generated takeoff schemas + manifest were written to"
    )
    schema_manifest_path: str | None = Field(None, description="Path to this run's schema manifest .json file")
    schema_files_written: list[str] = Field(
        default_factory=list, description="Schema .json files newly written by this run (excludes reused schemas)"
    )
