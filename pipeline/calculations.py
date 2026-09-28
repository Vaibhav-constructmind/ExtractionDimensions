"""Deterministic (non-LLM) engineering calculations over already-extracted
dimensions. Kept separate from orchestrator.py's extraction/correction logic
since these are pure arithmetic over numbers the model already produced, not
extraction post-processing.

Covers a generic, restricted-expression formula evaluator used for every
drawing's dynamic quantity takeoff: the model may propose a formula and the
raw inputs it needs, but every number in the output is computed here, never
by the model.
"""
from __future__ import annotations

import ast
import operator

from .schema import ComputedQuantity, DynamicQuantityItemSpec, MeasuredValue

# --- Unit conversion for dynamic-schema inputs ------------------------------
#
# A drawing's own dimensions are extracted in whatever unit is printed (mm,
# cm, m), but a formula's own convention (declared per-role via
# DynamicQuantityItemSpec.input_units) may need a different one for the same
# role -- most notably real rebar-weight formulas, which conventionally mix
# a millimetre bar diameter with a metre bar length in one expression. This
# is deliberately just a length-unit converter, not a "pick the right
# convention" heuristic: the model states which unit each role's formula
# needs, and this only ever converts a value FROM the unit it was actually
# found in TO that stated unit -- it never guesses a convention itself.

_LENGTH_UNITS_TO_METRES = {
    "mm": 0.001,
    "cm": 0.01,
    "m": 1.0,
}


class UnitConversionError(ValueError):
    """Raised when a value's unit or the target unit isn't a recognized
    length unit -- callers should leave the value unconverted (and let the
    caller decide whether that's still usable) rather than silently
    guessing a scale factor."""


def convert_length_unit(value: float, from_unit: str, to_unit: str) -> float:
    """Convert `value` from `from_unit` to `to_unit`, both length units
    (mm/cm/m). Returns `value` unchanged if `from_unit == to_unit` (no
    normalization risk from a no-op). Raises UnitConversionError for any
    unit not in the recognized length set -- e.g. a bare count or an
    already-derived area/volume/weight unit, which this function never
    touches since only individual length-basis inputs are converted before
    formula evaluation.
    """
    from_norm = (from_unit or "").strip().lower()
    to_norm = (to_unit or "").strip().lower()
    if from_norm == to_norm:
        return value
    if from_norm not in _LENGTH_UNITS_TO_METRES or to_norm not in _LENGTH_UNITS_TO_METRES:
        raise UnitConversionError(
            f"Cannot convert from {from_unit!r} to {to_unit!r} -- both must be one of "
            f"{sorted(_LENGTH_UNITS_TO_METRES)}"
        )
    return value * _LENGTH_UNITS_TO_METRES[from_norm] / _LENGTH_UNITS_TO_METRES[to_norm]


# --- Generic dynamic-schema formula evaluation -----------------------------
#
# A restricted arithmetic expression evaluator with NO `eval`/`exec` and no
# access to Python builtins, names, attributes, calls, or subscripts -- only
# numeric literals, the four arithmetic operators, exponentiation, unary
# +/-, parentheses, and lookups of names supplied explicitly by the caller.
# This is what lets the LLM safely *propose* a formula (e.g.
# 'wall_length * wall_height * wall_thickness') without ever being trusted
# to compute the actual number itself.

_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}
_ALLOWED_UNARYOPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


class FormulaError(ValueError):
    """Raised for an unparseable formula, a disallowed expression, or a
    missing variable -- always a caller-visible error, never a silent 0."""


def _eval_ast(node: ast.AST, variables: dict[str, float]) -> float:
    if isinstance(node, ast.Expression):
        return _eval_ast(node.body, variables)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return float(node.value)
        raise FormulaError(f"Unsupported constant in formula: {node.value!r}")
    if isinstance(node, ast.Name):
        if node.id not in variables:
            raise FormulaError(f"Unknown variable in formula: {node.id!r}")
        return float(variables[node.id])
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        left = _eval_ast(node.left, variables)
        right = _eval_ast(node.right, variables)
        return _ALLOWED_BINOPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARYOPS:
        return _ALLOWED_UNARYOPS[type(node.op)](_eval_ast(node.operand, variables))
    raise FormulaError(f"Disallowed expression in formula: {ast.dump(node)}")


def evaluate_formula(formula: str, variables: dict[str, float]) -> float:
    """Safely evaluate a restricted arithmetic `formula` (no eval/exec, no
    builtins, no attribute/subscript/call access -- literals, + - * / **
    // %, unary +/-, parentheses, and the supplied `variables` only).

    Raises FormulaError for anything outside that grammar, or for a
    variable name the formula references that isn't in `variables`.
    """
    try:
        tree = ast.parse(formula, mode="eval")
    except SyntaxError as exc:
        raise FormulaError(f"Formula is not a valid expression: {formula!r} ({exc})") from exc
    return _eval_ast(tree, variables)


def compute_dynamic_quantity(
    item: DynamicQuantityItemSpec,
    resolved_inputs: dict[str, tuple[float, str | None]],
) -> ComputedQuantity:
    """Compute one dynamic quantity item from its formula and already-
    resolved inputs.

    `resolved_inputs` maps each of `item.depends_on`'s role names to
    (value, source_dimension_id) -- matching a role name to an actual
    dimension on a specific drawing is the caller's job (see
    pipeline.handlers), since that's a per-drawing, classification-specific
    concern, not a generic arithmetic one. Any role in `item.depends_on`
    missing from `resolved_inputs` produces a null value with a `reason`,
    never a guessed number.
    """
    missing = [role for role in item.depends_on if role not in resolved_inputs]
    if missing:
        return ComputedQuantity(
            name=item.name,
            unit=item.unit,
            measurement_basis=item.measurement_basis,
            resource_category=item.resource_category,
            formula=item.formula,
            value=None,
            sources=[],
            reason=(
                "could not compute -- no matching dimension found on this drawing for: "
                + ", ".join(missing)
            ),
        )

    variables = {role: value for role, (value, _source) in resolved_inputs.items()}
    try:
        result = evaluate_formula(item.formula, variables)
    except FormulaError as exc:
        return ComputedQuantity(
            name=item.name,
            unit=item.unit,
            measurement_basis=item.measurement_basis,
            resource_category=item.resource_category,
            formula=item.formula,
            value=None,
            sources=[],
            reason=f"could not compute -- {exc}",
        )

    sources = [source for _value, source in resolved_inputs.values() if source]
    return ComputedQuantity(
        name=item.name,
        unit=item.unit,
        measurement_basis=item.measurement_basis,
        resource_category=item.resource_category,
        formula=item.formula,
        value=MeasuredValue(value=result, unit=item.unit),
        sources=sources,
        reason=None,
    )
