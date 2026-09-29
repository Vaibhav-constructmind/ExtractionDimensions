"""Cost-estimation rates for Claude calls made via Azure AI Foundry.

Rates are USD per million tokens, matching Anthropic's published list pricing
(Foundry billing may differ slightly by contract) -- this gives visibility
into relative/rough spend per drawing/PDF, not an exact invoice
reconciliation. Update _RATES_USD_PER_MTOK when pricing changes or a new
model is added; _DEFAULT_RATE_USD_PER_MTOK is used for any deployment name
not listed here, so an unrecognized/renamed model still gets a rough
estimate instead of crashing the whole run.
"""
from __future__ import annotations

# model name -> (input $ / Mtok, output $ / Mtok)
_RATES_USD_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-sonnet-5": (2.0, 10.0),
    "claude-opus-5-5": (15.0, 75.0),
    "claude-haiku-4-5-20251001": (1.0, 5.0),
    "claude-fable-5-1": (3.0, 15.0),
}
_DEFAULT_RATE_USD_PER_MTOK = (2.0, 10.0)


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """Rough USD cost for one call's token usage, per the rate table above
    (or the default rate if `model` isn't a recognized deployment name)."""
    input_rate, output_rate = _RATES_USD_PER_MTOK.get(model, _DEFAULT_RATE_USD_PER_MTOK)
    return (input_tokens / 1_000_000) * input_rate + (output_tokens / 1_000_000) * output_rate
