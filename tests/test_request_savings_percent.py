"""Tests for the mean of each request's own saving percentage on session and project rows."""

import json
from datetime import datetime, timezone

from headroom.proxy.savings_tracker import SavingsTracker

UTC = timezone.utc

# Two requests whose own savings differ: one saves 10 of 100 (10%), the other 0 of 1000 (0%). The mean of the two is 5%, while the overall figure (10 saved of 1100 before) is far lower.
SMALL_SAVING = {"input_tokens": 90, "tokens_saved": 10}
LARGE_NO_SAVING = {"input_tokens": 1000, "tokens_saved": 0}


def _record_two(tracker: SavingsTracker) -> None:
    tracker.record_request(model="gpt-4o", session_id="s", project="p", **SMALL_SAVING)
    tracker.record_request(model="gpt-4o", session_id="s", project="p", **LARGE_NO_SAVING)


def test_session_row_carries_the_mean_of_per_request_percentages_beside_the_overall_one(tmp_path):
    tracker = SavingsTracker(path=str(tmp_path / "savings.json"))
    _record_two(tracker)

    row = tracker.session_response("s")

    assert row is not None
    # Every request weighs the same: (10% + 0%) / 2.
    assert row["avg_request_savings_percent"] == 5.0
    # The overall figure weighs by size: 10 saved over 10 + 90 + 0 + 1000 before.
    assert row["savings_percent"] == 0.91


def test_project_row_carries_the_same_mean_as_the_session_row(tmp_path):
    tracker = SavingsTracker(path=str(tmp_path / "savings.json"))
    _record_two(tracker)

    project_row = tracker.lifetime_response()["projects"]["p"]

    assert project_row["avg_request_savings_percent"] == 5.0


def test_the_accumulators_survive_a_reload(tmp_path):
    path = tmp_path / "savings.json"
    _record_two(SavingsTracker(path=str(path)))

    reloaded = SavingsTracker(path=str(path))

    row = reloaded.session_response("s")
    assert row is not None
    assert row["avg_request_savings_percent"] == 5.0
    # A request after the reload extends the same mean: (10% + 0% + 20%) / 3.
    reloaded.record_request(
        model="gpt-4o", session_id="s", project="p", input_tokens=80, tokens_saved=20
    )
    extended = reloaded.session_response("s")
    assert extended is not None
    assert extended["avg_request_savings_percent"] == 10.0


def test_an_entry_persisted_before_the_accumulators_has_no_mean_until_requests_contribute(tmp_path):
    path = tmp_path / "savings.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 6,
                "lifetime": {},
                "display_session": None,
                "history": [],
                "sessions": {
                    "legacy": {
                        "requests": 500,
                        "tokens_saved": 5000,
                        "total_input_tokens": 100000,
                        "last_activity_at": datetime.now(UTC).isoformat(),
                    }
                },
            }
        )
    )
    tracker = SavingsTracker(path=str(path))

    legacy = tracker.session_response("legacy")
    assert legacy is not None
    # The 500 earlier requests contributed no percentages, so there is no mean to report, rather than one dragged down by requests it never saw.
    assert legacy["avg_request_savings_percent"] is None

    tracker.record_request(model="gpt-4o", session_id="legacy", input_tokens=90, tokens_saved=10)
    resumed = tracker.session_response("legacy")
    assert resumed is not None
    # Only the one request that has contributed counts, not the 501 the entry has seen.
    assert resumed["avg_request_savings_percent"] == 10.0
