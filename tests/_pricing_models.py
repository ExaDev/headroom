"""Pick a model litellm actually prices, instead of hardcoding one it may retire.

``litellm.model_cost`` is downloaded from GitHub at import time, so it is live
third-party data. BerriAI prunes retired models from it: on 2026-09-23
``claude-sonnet-4-20250514`` disappeared and every test that priced it began
failing with ``KeyError: 'input_cost_per_token'``, on every open pull request
at once, with no change on our side. Later the first-party ``claude-*`` entries
lost their ``*_above_200k_tokens`` tier fields altogether (Anthropic ended the
1M-context beta that carried that premium), so a hardcoded list of first-party
ids cannot be refreshed to satisfy a test that needs a long-context tier: the
only ``claude`` entries still carrying it are the Bedrock and Vertex AI rows.

The model id in those tests is incidental. They assert that Headroom's cost
arithmetic agrees with litellm's numbers, not that any particular model is
priced correctly, so the helper asks the catalog for *a* Claude model carrying
the fields the test needs, preferring Anthropic's own rows, then ids with no
provider or region qualifier, then alphabetical order, so selection is deterministic and needs no
upkeep as ids retire.

Pinning to the copy of the table vendored in the litellm wheel is not the
alternative it looks like: the two maps are complementary, not ordered. The
vendored map keeps retired ids but predates current models
(``claude-sonnet-5``), and its older entries lack newer fields such as
``input_cost_per_token_above_200k_tokens`` entirely.
"""

from __future__ import annotations

import pytest

_BASE_FIELDS = ("input_cost_per_token", "output_cost_per_token")

_FIRST_PARTY_PROVIDER = "anthropic"


def _candidate_order(item: tuple[str, dict[str, object]]) -> tuple[bool, bool, str]:
    """Sort key: first-party rows, then unqualified ids, then alphabetical."""
    key, info = item
    return (
        info.get("litellm_provider") != _FIRST_PARTY_PROVIDER,
        "/" in key or "." in key,
        key,
    )


def anthropic_pricing_model(*required_fields: str) -> str:
    """Return a currently-priced Claude model carrying ``required_fields``.

    Always includes the base input/output costs. Raises with an actionable
    message rather than skipping: if litellm prices no Claude model with these
    fields, the pricing tests are not measuring anything and that should be
    loud.
    """
    import litellm

    needed = set(_BASE_FIELDS) | set(required_fields)
    matches = [
        (key, info)
        for key, info in litellm.model_cost.items()
        if "claude" in key and isinstance(info, dict) and needed <= set(info)
    ]
    if not matches:
        raise AssertionError(
            f"litellm prices no Claude model with the fields {sorted(needed)}. "
            "It most likely dropped them from model_prices_and_context_window.json; "
            "check which fields current Claude rows carry before changing the test."
        )
    return min(matches, key=_candidate_order)[0]


@pytest.fixture(scope="session")
def anthropic_model() -> str:
    """Session fixture wrapper for tests that prefer injection over a constant."""
    return anthropic_pricing_model()
