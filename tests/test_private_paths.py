"""The rule deciding whether another user could change a path the proxy trusts."""

from __future__ import annotations

import os
import stat

import pytest

from headroom.proxy.private_paths import foreign_write_access

pytestmark = pytest.mark.skipif(not hasattr(os, "geteuid"), reason="POSIX ownership only")

ME = 1000
ROOT = 0
OTHER = 2000


def _stat(uid: int, mode: int) -> os.stat_result:
    return os.stat_result((mode, 0, 0, 1, uid, 0, 0, 0, 0, 0))


@pytest.fixture(autouse=True)
def _running_as_me(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, "geteuid", lambda: ME)


FILE = stat.S_IFREG
DIR = stat.S_IFDIR


@pytest.mark.parametrize(
    ("info", "kwargs"),
    [
        (_stat(ME, FILE | 0o600), {}),
        (_stat(ME, DIR | 0o700), {}),
        (_stat(ROOT, FILE | 0o644), {"root_may_own": True}),
        (_stat(ROOT, DIR | 0o755), {"root_may_own": True}),
    ],
    ids=[
        "own-file",
        "own-dir",
        "root-file",
        "root-dir-0755",
    ],
)
def test_accepted(info: os.stat_result, kwargs: dict[str, bool]) -> None:
    assert foreign_write_access(info, **kwargs) is None


@pytest.mark.parametrize(
    ("info", "kwargs", "description"),
    [
        (
            _stat(ROOT, DIR | 0o700),
            {},
            "owned by uid 0, not by the user running the proxy (uid 1000)",
        ),
        (
            _stat(OTHER, FILE | 0o600),
            {"root_may_own": True},
            "owned by uid 2000, not by the user running the proxy (uid 1000) or root",
        ),
        (
            _stat(ME, FILE | 0o620),
            {"root_may_own": True},
            "writable by group or others (mode 0620)",
        ),
        (
            _stat(ME, FILE | 0o1666),
            {"root_may_own": True},
            "writable by group or others (mode 1666)",
        ),
        (_stat(ME, DIR | 0o1777), {}, "writable by group or others (mode 1777)"),
        (
            _stat(ROOT, DIR | 0o1777),
            {"root_may_own": True},
            "writable by group or others (mode 1777)",
        ),
        (
            _stat(ME, DIR | 0o1777),
            {"root_may_own": True},
            "writable by group or others (mode 1777)",
        ),
    ],
    ids=[
        "uds-root-dir",
        "other-owner-file",
        "group-writable-file",
        "sticky-file-not-shared",
        "uds-sticky-dir",
        "root-sticky-dir",
        "own-sticky-dir",
    ],
)
def test_refused(info: os.stat_result, kwargs: dict[str, bool], description: str) -> None:
    problem = foreign_write_access(info, **kwargs)
    assert problem is not None
    assert problem.description == description
