"""Unit tests for pipeline/pricing.py's cost-estimation rate lookup.

Run with:
    .venv\\Scripts\\python.exe -m unittest tests.test_pricing -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.pricing import estimate_cost_usd


class TestEstimateCostUsd(unittest.TestCase):
    def test_known_model_uses_its_own_rate(self):
        cost = estimate_cost_usd("claude-sonnet-5", 1_000_000, 0)
        self.assertAlmostEqual(cost, 3.0)

    def test_output_tokens_priced_separately(self):
        cost = estimate_cost_usd("claude-sonnet-5", 0, 1_000_000)
        self.assertAlmostEqual(cost, 15.0)

    def test_zero_tokens_is_zero_cost(self):
        self.assertEqual(estimate_cost_usd("claude-sonnet-5", 0, 0), 0.0)

    def test_unrecognized_model_falls_back_to_default_rate(self):
        cost = estimate_cost_usd("some-future-model-nobody-added-yet", 1_000_000, 1_000_000)
        self.assertAlmostEqual(cost, 3.0 + 15.0)

    def test_mixed_tokens_sum_correctly(self):
        cost = estimate_cost_usd("claude-haiku-4-5-20251001", 500_000, 200_000)
        self.assertAlmostEqual(cost, 0.5 * 1.0 + 0.2 * 5.0)


if __name__ == "__main__":
    unittest.main()
