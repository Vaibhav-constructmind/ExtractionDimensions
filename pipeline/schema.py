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


class MeasuredValue(BaseModel):
    """A single numeric value with its unit -- used as an auditable input or
    output of a deterministic (non-LLM) calculation, as opposed to
    QuantityField which also carries a model-derived confidence/method."""

    value: float
    unit: str
    source: str | None = Field(
        None, description="dimension_id this value came from, if traceable -- never fabricated"
    )


class SingleStepConcrete(BaseModel):
    """Concrete volume of ONE stair step, computed deterministically (never
    by the model) as a triangular prism: 0.5 * tread_depth * riser_height *
    stair_width. Covers exactly one step -- per-flight volume, landing
    volume, and total stair/project volume are separate, not-yet-implemented
    calculations.
    """

    tread_depth: MeasuredValue
    riser_height: MeasuredValue
    stair_width: MeasuredValue
    volume: MeasuredValue = Field(..., description="0.5 * tread_depth * riser_height * stair_width, in m3")
    formula: str = Field(..., description="The formula as evaluated, e.g. '0.5 × 0.290 × 0.165 × 1.570'")
    tread_depth_source: str | None = Field(
        None, description="dimension_id this tread_depth came from, if traceable"
    )
    riser_height_source: str | None = Field(
        None, description="dimension_id this riser_height came from, if traceable"
    )
    stair_width_source: str | None = Field(
        None, description="dimension_id this stair_width came from, if traceable"
    )


class QuantityTakeoff(BaseModel):
    """Stair/enclosure quantity-takeoff figures derived from this drawing's dimensions and datums.

    Vertical/stair-geometry fields (flights, risers, treads, landings, vertical drop) are
    typically derivable from a section view's stair_rise dimensions and elevation_datums.
    Wall/footprint/concrete/formwork fields normally require a plan view (footprint,
    wall centerlines, thickness) and are left null with an explanatory note when the
    source drawing doesn't show them.
    """

    num_doors: QuantityField | None = Field(None, description="Count of doors visible on this drawing")
    num_drains: QuantityField | None = Field(None, description="Count of drains visible on this drawing")
    num_flights: QuantityField | None = None
    num_risers_total: QuantityField | None = None
    riser_height: QuantityField | None = None
    single_step_concrete: SingleStepConcrete | None = Field(
        None,
        description=(
            "Concrete volume of ONE stair step, computed deterministically in Python "
            "(not by the model) from this drawing's own tread/riser/width dimensions."
        ),
    )
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


class IncompleteFlight(BaseModel):
    """A step count found on one side (tread or riser) of a stair group with
    no matching reading on the other side, so no volume could be computed
    for it without fabricating the missing value."""

    num_steps: int
    reason: str


class FlightConcrete(BaseModel):
    """Concrete volume for ONE physical stair flight on ONE drawing (a run
    of steps sharing the same tread depth, riser height, and width), as a
    triangular prism per step:

        volume_per_step = 0.5 * tread_depth * riser_height * stair_width
        total_volume = volume_per_step * num_steps

    Every flight found on a drawing gets its OWN record here -- flights are
    never pooled or copied across drawings, and two physical flights that
    happen to share a step count still each get their own entry (no
    instance-count collapsing).
    """

    flight_label: str = Field(
        ...,
        description=(
            "This flight's physical position within its drawing, e.g. 'upper_flight' or "
            "'lower_flight' (read from the drawing's own annotations when they say so, "
            "otherwise an ordinal position like 'flight_1') -- never the step count."
        ),
    )
    num_steps: MeasuredValue = Field(..., description="Step count for this flight (unit='count')")
    num_steps_method: str | None = Field(
        None, description="How num_steps was resolved, e.g. a tread/riser count disagreement"
    )
    tread_depth: MeasuredValue
    riser_height: MeasuredValue
    stair_width: MeasuredValue = Field(..., description="This flight's OWN clear width -- may differ from another flight's on the same drawing")
    volume_per_step: MeasuredValue = Field(..., description="0.5 * tread_depth * riser_height * stair_width, in m3")
    formula_per_step: str
    total_volume: MeasuredValue = Field(..., description="volume_per_step * num_steps, in m3")
    formula_total: str


class StairQuantityTakeoff(BaseModel):
    """Stair concrete-volume takeoff for ONE drawing, computed entirely from
    that drawing's OWN dimensions -- tread, riser, and stair width are never
    pooled or copied from a sibling drawing in the same stair group.
    `stair_group_id` identifies which physical stair this drawing belongs to
    (drawings sharing a title stem like 'STAIR-07-*'); it's identity only,
    not a dimension pool. Computed entirely in Python (pipeline.calculations)
    from dimensions the model already extracted and tagged; the model never
    performs this arithmetic.
    """

    stair_group_id: str = Field(..., description="e.g. 'STAIR-07', the physical stair this drawing belongs to")
    source_drawings: list[str] = Field(
        default_factory=list,
        description="Always just this one drawing's drawing_id -- kept as a list for schema stability.",
    )
    flights: list[FlightConcrete] = Field(
        default_factory=list,
        description="One entry per physical flight actually found on this drawing -- the authoritative, exact per-flight volumes",
    )
    incomplete_flights: list[IncompleteFlight] = Field(
        default_factory=list,
        description="Step counts found on only one side (tread, riser, or width) on this drawing -- recorded rather than fabricated",
    )
    tread_depth_representative: MeasuredValue = Field(
        ..., description="Most common tread depth across this drawing's resolved flights -- reference only; precise values are in `flights[]`"
    )
    riser_height_representative: MeasuredValue = Field(
        ..., description="Mode riser height across this drawing's resolved flights -- reference only; precise values are in `flights[]`"
    )
    riser_height_method: str = Field(..., description="How the representative riser height was chosen")
    stair_width_representative: MeasuredValue = Field(
        ..., description="Mode stair width across this drawing's resolved flights -- reference only; each flight's own stair_width in `flights[]` is authoritative"
    )
    num_steps_total: MeasuredValue = Field(..., description="Sum of num_steps across this drawing's own flights (unit='count')")
    total_volume: MeasuredValue = Field(..., description="Sum of this drawing's own flights' total_volume -- exact, not approximated")
    formula_total: str = Field(default="sum of each flight's total_volume across this drawing's own flights")


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
    quantity_takeoff: QuantityTakeoff | None = Field(
        default=None, description="Stair/enclosure quantities derived from this drawing's dimensions and datums"
    )
    stair_quantity_takeoff: StairQuantityTakeoff | None = Field(
        default=None,
        description=(
            "Stair concrete-volume takeoff for THIS drawing's own stair flights, computed "
            "solely from this drawing's own dimensions -- never pooled or copied from another "
            "drawing in the same stair group."
        ),
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
