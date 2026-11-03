"""`header_safe_transforms` keeps the comma-joined transforms header splittable.

`x-headroom-transforms` is built as ``",".join(transforms_applied)``. The
enriched ``read_lifecycle:<state>:<path>`` and ``smart_crush:<n>:<names>`` tags
can contain commas, which would make that header ambiguous; the helper collapses
them back to their legacy counter shape so each header token stays comma-free.
"""

from __future__ import annotations

from headroom.proxy.cost import (
    TRANSFORMS_HEADER_MAX_BYTES,
    header_safe_transforms,
    transforms_header_value,
)


def test_strips_smart_crush_tool_names():
    assert header_safe_transforms(["smart_crush:2:Bash,Grep"]) == ["smart_crush:2"]


def test_strips_read_lifecycle_path():
    assert header_safe_transforms(["read_lifecycle:stale:/src/App.tsx"]) == ["read_lifecycle:stale"]


def test_strips_read_lifecycle_path_with_comma():
    # A path containing a comma is exactly the case that would corrupt the header.
    assert header_safe_transforms(["read_lifecycle:superseded:/tmp/a,b/x.py"]) == [
        "read_lifecycle:superseded"
    ]


def test_passes_through_legacy_and_unrelated_tags():
    tags = ["smart_crush:3", "read_lifecycle:stale", "router:excluded:tool", "smart:lossless:table"]
    assert header_safe_transforms(tags) == tags


def test_joined_header_remains_unambiguous():
    tags = ["smart_crush:2:Bash,Grep", "read_lifecycle:stale:/a,b.py", "router:excluded:tool"]
    header = ",".join(header_safe_transforms(tags))
    # One token per tag — no stray commas leaking in from enriched detail.
    assert header.split(",") == ["smart_crush:2", "read_lifecycle:stale", "router:excluded:tool"]


# --- transforms_header_value: the header stays inside a fixed byte budget ---


def test_header_value_matches_the_plain_join_when_nothing_repeats_and_it_fits():
    tags = ["smart_crush:3", "read_lifecycle:stale", "router:excluded:tool"]
    assert transforms_header_value(tags) == ",".join(tags)


def test_header_value_collapses_repeats_to_a_counted_tag_in_first_seen_order():
    tags = ["router:excluded:tool"] * 3 + ["smart_crush:2"] + ["router:excluded:tool"]
    assert transforms_header_value(tags) == "router:excluded:tool*4,smart_crush:2"


def test_header_value_of_a_huge_repetitive_list_is_small_and_loses_nothing():
    tags = ["router:excluded:tool", "cache_hit", "lossless_noop"] * 5000
    value = transforms_header_value(tags)
    assert value.split(",") == ["router:excluded:tool*5000", "cache_hit*5000", "lossless_noop*5000"]


def test_header_value_never_exceeds_the_budget_and_reports_what_it_dropped():
    # More distinct tags than fit: every tag is unique, so collapsing cannot help.
    tags = [f"router:excluded:distinct_tag_number_{i}" for i in range(2000)]
    value = transforms_header_value(tags)
    assert len(value.encode()) <= TRANSFORMS_HEADER_MAX_BYTES
    kept, marker = value.rsplit(",", 1)
    assert marker.startswith("truncated:")
    dropped = int(marker.removeprefix("truncated:"))
    assert len(kept.split(",")) + dropped == len(tags)
    # What survives is a prefix of the input, in order.
    assert kept.split(",") == tags[: len(kept.split(","))]


def test_header_value_budget_is_measured_in_bytes_not_characters():
    tags = [f"tag:ééé{i}" for i in range(2000)]
    assert len(transforms_header_value(tags).encode()) <= TRANSFORMS_HEADER_MAX_BYTES


def test_header_value_budget_sits_below_the_smallest_common_header_block_limit():
    # 8 KiB is the default header buffer of common servers and proxies; one header must leave the
    # rest of the block (upstream passthrough headers, the other x-headroom-* values) room under it.
    assert TRANSFORMS_HEADER_MAX_BYTES <= 8192 // 2
