"""Tests for HEADROOM_ALLOWED_BASE_URLS_FILE, the allowlist the guard re-reads at runtime.

A supervisor that lists its own loopback endpoint cannot restart a proxy that sessions are still using when that endpoint's port changes, so the guard has to pick up a rewritten file on the next request. Every case uses IP literals, so no external network is needed.
"""

from __future__ import annotations

import errno
import logging
import os
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner, Result
from fastapi.testclient import TestClient

from headroom.cli.main import main
from headroom.proxy import upstream_guard
from headroom.proxy.server import ProxyConfig, create_app
from headroom.proxy.upstream_guard import is_safe_upstream_url
from headroom.proxy.upstream_pinning import proxied_upstream_listed

FILE_ENV = "HEADROOM_ALLOWED_BASE_URLS_FILE"
LIST_ENV = "HEADROOM_ALLOWED_BASE_URLS"


def _replace(path: Path, text: str) -> None:
    """Rewrite ``path`` the way a supervisor should: a sibling, then a rename."""
    staging = path.with_name(path.name + ".tmp")
    staging.write_text(text, encoding="utf-8", newline="\n")
    os.replace(staging, path)


def _count_reads(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record every path the guard actually opens and reads from here on."""
    reads: list[str] = []
    real_read = upstream_guard._read_allowlist_file

    def counting_read(path: str, st: os.stat_result):  # noqa: ANN202
        reads.append(path)
        return real_read(path, st)

    monkeypatch.setattr(upstream_guard, "_read_allowlist_file", counting_read)
    return reads


@pytest.fixture
def allow_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "allowed-base-urls"
    monkeypatch.setenv(FILE_ENV, str(path))
    return path


def test_a_listed_origin_is_admitted_and_others_are_refused(allow_file: Path) -> None:
    _replace(allow_file, "http://127.0.0.1:4100\n")

    assert is_safe_upstream_url("http://127.0.0.1:4100/v1") is True
    # Exact origin only, and allowlist mode refuses public hosts too.
    assert is_safe_upstream_url("http://127.0.0.1:4101/v1") is False
    assert is_safe_upstream_url("https://8.8.8.8/v1") is False


def test_a_replaced_file_is_picked_up_without_a_restart(allow_file: Path) -> None:
    _replace(allow_file, "http://127.0.0.1:4100")
    assert is_safe_upstream_url("http://127.0.0.1:4200/v1") is False

    _replace(allow_file, "http://127.0.0.1:4200")

    assert is_safe_upstream_url("http://127.0.0.1:4200/v1") is True
    assert is_safe_upstream_url("http://127.0.0.1:4100/v1") is False


def test_an_unchanged_file_is_not_read_again(
    allow_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _replace(allow_file, "http://127.0.0.1:4100")
    reads = _count_reads(monkeypatch)

    for _ in range(5):
        assert is_safe_upstream_url("http://127.0.0.1:4100/v1") is True
    assert len(reads) == 1

    _replace(allow_file, "http://127.0.0.1:4100,http://127.0.0.1:4300")
    assert is_safe_upstream_url("http://127.0.0.1:4300/v1") is True
    assert len(reads) == 2


def test_commas_and_newlines_both_separate_entries(allow_file: Path) -> None:
    _replace(allow_file, "gateway.internal, http://127.0.0.1:4100\r\nhttps://llm.corp:8443\n\n")

    assert is_safe_upstream_url("https://gateway.internal/v1") is True
    assert is_safe_upstream_url("http://127.0.0.1:4100/v1") is True
    assert is_safe_upstream_url("https://llm.corp:8443/v1") is True


def test_file_entries_are_merged_with_the_variable(
    allow_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(LIST_ENV, "gateway.internal")
    _replace(allow_file, "http://127.0.0.1:4100")

    assert is_safe_upstream_url("https://gateway.internal/v1") is True
    assert is_safe_upstream_url("http://127.0.0.1:4100/v1") is True
    assert is_safe_upstream_url("https://8.8.8.8/v1") is False


def test_an_empty_file_is_an_empty_allowlist_not_the_default_policy(
    allow_file: Path,
) -> None:
    """Truncating the file must refuse everything, never admit every public host."""
    _replace(allow_file, "")

    assert is_safe_upstream_url("https://8.8.8.8/v1") is False
    assert is_safe_upstream_url("http://127.0.0.1:4100/v1") is False


def test_a_missing_file_fails_closed_and_is_logged_once(
    allow_file: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING, logger="headroom.proxy.upstream_guard")

    for _ in range(3):
        assert is_safe_upstream_url("https://8.8.8.8/v1") is False
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert FILE_ENV in warnings[0].getMessage()
    assert str(allow_file) in warnings[0].getMessage()

    # Once it can be read again it is honoured, and a later failure is a new one worth reporting.
    _replace(allow_file, "http://127.0.0.1:4100")
    assert is_safe_upstream_url("http://127.0.0.1:4100/v1") is True
    allow_file.unlink()
    assert is_safe_upstream_url("http://127.0.0.1:4100/v1") is False
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 2


def test_an_unreadable_file_keeps_only_the_variable(
    allow_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(LIST_ENV, "gateway.internal")
    allow_file.mkdir()  # exists, but cannot be read as a file

    assert is_safe_upstream_url("https://gateway.internal/v1") is True
    assert is_safe_upstream_url("http://127.0.0.1:4100/v1") is False
    assert is_safe_upstream_url("https://8.8.8.8/v1") is False


def test_a_file_that_is_not_utf8_fails_closed(allow_file: Path) -> None:
    allow_file.write_bytes(b"http://127.0.0.1:4100\xff\n")

    assert is_safe_upstream_url("http://127.0.0.1:4100/v1") is False
    assert is_safe_upstream_url("https://8.8.8.8/v1") is False


def test_entries_from_an_earlier_version_are_dropped_when_the_file_breaks(
    allow_file: Path,
) -> None:
    _replace(allow_file, "http://127.0.0.1:4100")
    assert is_safe_upstream_url("http://127.0.0.1:4100/v1") is True

    allow_file.write_bytes(b"\xff")

    assert is_safe_upstream_url("http://127.0.0.1:4100/v1") is False


def test_a_same_size_rewrite_with_its_mtime_restored_is_noticed(allow_file: Path) -> None:
    """``cp -p``, ``touch -r`` and coarse timestamps all leave mtime and size unchanged."""
    allow_file.write_text("http://127.0.0.1:4100", encoding="utf-8")
    assert is_safe_upstream_url("http://127.0.0.1:4100/v1") is True
    before = allow_file.stat()

    with allow_file.open("r+", encoding="utf-8") as handle:
        handle.write("http://127.0.0.1:4200")
    os.utime(allow_file, ns=(before.st_atime_ns, before.st_mtime_ns))
    after = allow_file.stat()
    assert (after.st_ino, after.st_size, after.st_mtime_ns) == (
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )

    assert is_safe_upstream_url("http://127.0.0.1:4200/v1") is True
    assert is_safe_upstream_url("http://127.0.0.1:4100/v1") is False


@pytest.mark.skipif(
    not hasattr(os, "geteuid") or os.geteuid() == 0,
    reason="needs POSIX permissions that bind the current user (root reads mode 000 files)",
)
def test_revoking_read_permission_on_a_cached_file_takes_effect(
    allow_file: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING, logger="headroom.proxy.upstream_guard")
    _replace(allow_file, "http://127.0.0.1:4100")
    assert is_safe_upstream_url("http://127.0.0.1:4100/v1") is True

    allow_file.chmod(0)
    try:
        assert is_safe_upstream_url("http://127.0.0.1:4100/v1") is False
        assert is_safe_upstream_url("http://127.0.0.1:4100/v1") is False
    finally:
        allow_file.chmod(0o600)

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "Permission denied" in warnings[0].getMessage()


_MALFORMED = pytest.mark.parametrize(
    "bad",
    [
        "[oops",  # unbalanced IPv6 bracket, bare
        "http://[::1",  # unbalanced IPv6 bracket in a URL
        "http://a℀b.example",  # NFKC turns U+2100 into "a/c", URL syntax in the netloc
        "a℀b.example",  # the same, bare
    ],
)


@_MALFORMED
def test_a_malformed_file_entry_is_skipped_not_fatal(
    allow_file: Path, bad: str, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="headroom.proxy.upstream_guard")
    _replace(allow_file, f"{bad}\nhttp://127.0.0.1:4100\n")

    assert is_safe_upstream_url("http://127.0.0.1:4100/v1") is True
    assert is_safe_upstream_url("https://8.8.8.8/v1") is False
    loads = [r.getMessage() for r in caplog.records if "loaded" in r.getMessage()]
    assert len(loads) == 1
    assert "1 invalid entry skipped" in loads[0]


@_MALFORMED
def test_a_malformed_variable_entry_is_skipped_not_fatal(
    monkeypatch: pytest.MonkeyPatch, bad: str, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING, logger="headroom.proxy.upstream_guard")
    monkeypatch.setenv(LIST_ENV, f"{bad},gateway.internal")

    for _ in range(3):
        assert is_safe_upstream_url("https://gateway.internal/v1") is True
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings == [f"{LIST_ENV}: skipped 1 invalid entry"]


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs FIFOs")
def test_a_fifo_at_the_path_is_refused_without_blocking(
    allow_file: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Opening a FIFO for reading blocks until a writer appears; the check must not."""
    caplog.set_level(logging.WARNING, logger="headroom.proxy.upstream_guard")
    monkeypatch.setenv(LIST_ENV, "gateway.internal")
    os.mkfifo(allow_file)
    results: list[bool] = []

    def check() -> None:
        results.append(is_safe_upstream_url("http://127.0.0.1:4100/v1"))
        results.append(is_safe_upstream_url("https://gateway.internal/v1"))

    worker = threading.Thread(target=check, daemon=True)
    worker.start()
    worker.join(timeout=5)

    if worker.is_alive():
        # Release the blocked open so the daemon thread does not outlive the test.
        with open(allow_file, "w", encoding="utf-8"):
            pass
        pytest.fail("the check blocked opening a FIFO")
    assert results == [False, True]
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "not a regular file" in warnings[0]


def test_an_oversized_file_is_refused_whole_not_truncated(
    allow_file: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Its first bytes alone would be a valid list; the file as a whole is not one."""
    caplog.set_level(logging.WARNING, logger="headroom.proxy.upstream_guard")
    head = "http://127.0.0.1:4100\n"
    padding = "\n" * (upstream_guard._MAX_ALLOWLIST_FILE_BYTES + 1 - len(head))
    _replace(allow_file, head + padding)

    assert is_safe_upstream_url("http://127.0.0.1:4100/v1") is False
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert f"larger than {upstream_guard._MAX_ALLOWLIST_FILE_BYTES} bytes" in warnings[0]


def test_a_file_exactly_at_the_size_cap_is_read(allow_file: Path) -> None:
    head = "http://127.0.0.1:4100\n"
    _replace(allow_file, head + "\n" * (upstream_guard._MAX_ALLOWLIST_FILE_BYTES - len(head)))

    assert is_safe_upstream_url("http://127.0.0.1:4100/v1") is True


@pytest.mark.parametrize("bare", ["127.0.0.1:4100", "[::1]:4100", "gateway.internal:8443"])
def test_a_bare_host_with_a_port_is_invalid_in_the_file(
    allow_file: Path, bare: str, caplog: pytest.LogCaptureFixture
) -> None:
    """Read as a bare host it would admit every port on that host."""
    caplog.set_level(logging.INFO, logger="headroom.proxy.upstream_guard")
    _replace(allow_file, bare)
    host = bare.rsplit(":", 1)[0]

    assert is_safe_upstream_url(f"http://{host}:4100/v1") is False
    assert is_safe_upstream_url(f"http://{host}:9999/v1") is False
    assert any("1 invalid entry skipped" in r.getMessage() for r in caplog.records)


def test_a_bare_host_with_a_port_is_invalid_in_the_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(LIST_ENV, "127.0.0.1:4100,gateway.internal")

    assert is_safe_upstream_url("http://127.0.0.1:4100/v1") is False
    assert is_safe_upstream_url("http://127.0.0.1:9999/v1") is False
    assert is_safe_upstream_url("https://gateway.internal:8443/v1") is True


def test_a_byte_order_mark_is_ignored(allow_file: Path) -> None:
    allow_file.write_bytes("\ufeffhttp://127.0.0.1:4100\n".encode())

    assert is_safe_upstream_url("http://127.0.0.1:4100/v1") is True


def test_carriage_returns_and_tabs_separate_entries(allow_file: Path) -> None:
    allow_file.write_bytes(b"http://127.0.0.1:4100\rhttp://127.0.0.1:4200\thttp://127.0.0.1:4300")

    for port in (4100, 4200, 4300):
        assert is_safe_upstream_url(f"http://127.0.0.1:{port}/v1") is True


def test_whole_line_comments_are_ignored_in_the_file(
    allow_file: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="headroom.proxy.upstream_guard")
    _replace(
        allow_file,
        "# written by the supervisor\n  # gateway.internal, http://127.0.0.1:4200\nhttp://127.0.0.1:4100\n",
    )

    assert is_safe_upstream_url("http://127.0.0.1:4100/v1") is True
    assert is_safe_upstream_url("https://gateway.internal/v1") is False
    assert is_safe_upstream_url("http://127.0.0.1:4200/v1") is False
    assert any("0 invalid entries skipped" in r.getMessage() for r in caplog.records)


def test_a_hash_entry_in_the_variable_is_not_a_comment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(LIST_ENV, "#gateway.internal\nhttp://127.0.0.1:4100")

    assert is_safe_upstream_url("http://127.0.0.1:4100/v1") is True
    assert is_safe_upstream_url("https://gateway.internal/v1") is False


def test_the_identity_cached_is_the_one_read_not_the_one_first_stat_ed(
    allow_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rename between the guard's ``stat`` and its ``open`` must not cost a second read."""
    _replace(allow_file, "http://127.0.0.1:4100")
    real_read = upstream_guard._read_allowlist_file

    def replaced_after_stat(path: str, st: os.stat_result):  # noqa: ANN202
        _replace(allow_file, "http://127.0.0.1:4200")
        return real_read(path, st)

    monkeypatch.setattr(upstream_guard, "_read_allowlist_file", replaced_after_stat)
    assert is_safe_upstream_url("http://127.0.0.1:4200/v1") is True

    reads = _count_reads(monkeypatch)
    assert is_safe_upstream_url("http://127.0.0.1:4200/v1") is True
    assert reads == []


def _break_oversized(path: Path) -> None:
    path.write_text("\n" * (upstream_guard._MAX_ALLOWLIST_FILE_BYTES + 1), encoding="utf-8")


def _break_not_utf8(path: Path) -> None:
    path.write_bytes(b"http://127.0.0.1:4100\xff")


def _break_unreadable(path: Path) -> None:
    _replace(path, "http://127.0.0.1:4100")
    path.chmod(0)


def _break_fifo(path: Path) -> None:
    os.mkfifo(path)


def _break_directory(path: Path) -> None:
    path.mkdir()


def _remove(path: Path) -> None:
    if path.is_dir():
        path.rmdir()
    else:
        path.chmod(0o600)
        path.unlink()


@pytest.mark.parametrize(
    ("break_file", "opens_expected"),
    [
        pytest.param(_break_oversized, 1, id="oversized"),
        pytest.param(_break_not_utf8, 1, id="not-utf8"),
        pytest.param(
            _break_unreadable,
            1,
            id="unreadable",
            marks=pytest.mark.skipif(
                not hasattr(os, "geteuid") or os.geteuid() == 0,
                reason="needs POSIX permissions that bind the current user",
            ),
        ),
        pytest.param(
            _break_fifo,
            0,
            id="fifo",
            marks=pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs FIFOs"),
        ),
        pytest.param(_break_directory, 0, id="directory"),
    ],
)
def test_an_unusable_file_is_tried_once_per_version_and_a_repair_is_picked_up(
    allow_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    break_file: object,
    opens_expected: int,
) -> None:
    """A broken file costs a ``stat`` per check, not an open and a read on every request."""
    assert callable(break_file)
    break_file(allow_file)
    reads = _count_reads(monkeypatch)
    opens: list[str] = []
    real_open = os.open

    def counting_open(path: str, *args: object, **kwargs: object) -> int:
        if os.fspath(path) == str(allow_file):
            opens.append(os.fspath(path))
        return real_open(path, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(os, "open", counting_open)

    for _ in range(5):
        assert is_safe_upstream_url("http://127.0.0.1:4100/v1") is False
    assert len(reads) == 1
    assert len(opens) == opens_expected

    _remove(allow_file)
    _replace(allow_file, "http://127.0.0.1:4100")
    assert is_safe_upstream_url("http://127.0.0.1:4100/v1") is True
    assert len(reads) == 2


def test_an_oversized_file_is_refused_from_its_size_before_any_read(
    allow_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _break_oversized(allow_file)
    read_sizes: list[int] = []
    real_fdopen = os.fdopen

    class _RecordingHandle:
        def __init__(self, handle: object) -> None:
            self._handle = handle

        def __enter__(self) -> _RecordingHandle:
            return self

        def __exit__(self, *exc: object) -> None:
            self._handle.close()  # type: ignore[attr-defined]

        def fileno(self) -> int:
            return int(self._handle.fileno())  # type: ignore[attr-defined]

        def read(self, size: int = -1) -> bytes:
            read_sizes.append(size)
            return bytes(self._handle.read(size))  # type: ignore[attr-defined]

    monkeypatch.setattr(
        os, "fdopen", lambda fd, *a, **k: _RecordingHandle(real_fdopen(fd, *a, **k))
    )

    assert is_safe_upstream_url("http://127.0.0.1:4100/v1") is False
    assert read_sizes == []


@pytest.mark.parametrize("error", [errno.EMFILE, errno.ENFILE, errno.EIO, errno.ENOMEM])
def test_a_transient_open_failure_is_retried_on_the_next_check(
    allow_file: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, error: int
) -> None:
    """Running out of descriptors says nothing about the file, so it must not stick to it."""
    caplog.set_level(logging.WARNING, logger="headroom.proxy.upstream_guard")
    _replace(allow_file, "http://127.0.0.1:4100")
    real_open = os.open
    failures = [error, error]

    def failing_open(path: str, *args: object, **kwargs: object) -> int:
        if os.fspath(path) == str(allow_file) and failures:
            raise OSError(failures.pop(), os.strerror(error), os.fspath(path))
        return real_open(path, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(os, "open", failing_open)

    assert is_safe_upstream_url("http://127.0.0.1:4100/v1") is False
    assert is_safe_upstream_url("http://127.0.0.1:4100/v1") is False
    assert is_safe_upstream_url("http://127.0.0.1:4100/v1") is True
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert os.strerror(error) in warnings[0]


def test_the_failure_warning_says_what_each_route_does(
    allow_file: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Not every route answers 400: the OpenAI handlers fall back to the configured upstream."""
    caplog.set_level(logging.WARNING, logger="headroom.proxy.upstream_guard")

    assert is_safe_upstream_url("http://127.0.0.1:4100/v1") is False

    (warning,) = [r.getMessage() for r in caplog.records]
    assert "rejected with 400" not in warning
    assert "/v1/messages, /v1/alpha/search and the catch-all passthrough answer 400" in warning
    assert "send the request to the configured upstream" in warning


def test_the_openai_chat_handler_ignores_a_refused_override(allow_file: Path) -> None:
    """The behaviour the warning describes, read from the handler rather than assumed."""
    from headroom.proxy.handlers.openai import _resolve_openai_upstream_base

    _replace(allow_file, "")

    assert _resolve_openai_upstream_base({"x-headroom-base-url": "http://127.0.0.1:4100"}) is None


def test_the_guard_expands_a_home_relative_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    upstream_guard.resolve_allowlist_file_path.cache_clear()
    _replace(tmp_path / "allowed", "http://127.0.0.1:4100")
    monkeypatch.setenv(FILE_ENV, "~/allowed")

    assert is_safe_upstream_url("http://127.0.0.1:4100/v1") is True


def test_a_relative_path_keeps_the_directory_of_its_first_use(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    upstream_guard.resolve_allowlist_file_path.cache_clear()
    first, elsewhere = tmp_path / "first", tmp_path / "elsewhere"
    first.mkdir()
    elsewhere.mkdir()
    _replace(first / "allowed", "http://127.0.0.1:4100")
    _replace(elsewhere / "allowed", "http://127.0.0.1:4200")
    monkeypatch.setenv(FILE_ENV, "allowed")
    monkeypatch.chdir(first)
    assert is_safe_upstream_url("http://127.0.0.1:4100/v1") is True

    monkeypatch.chdir(elsewhere)

    assert is_safe_upstream_url("http://127.0.0.1:4100/v1") is True
    assert is_safe_upstream_url("http://127.0.0.1:4200/v1") is False


_POSIX_USER = pytest.mark.skipif(
    not hasattr(os, "geteuid") or os.geteuid() == 0,
    reason="needs POSIX ownership and modes that bind the current user",
)


@_POSIX_USER
def test_a_group_writable_file_is_refused(
    allow_file: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING, logger="headroom.proxy.upstream_guard")
    _replace(allow_file, "http://127.0.0.1:4100")
    allow_file.chmod(0o620)

    assert is_safe_upstream_url("http://127.0.0.1:4100/v1") is False
    assert is_safe_upstream_url("http://127.0.0.1:4100/v1") is False
    (warning,) = [r.getMessage() for r in caplog.records]
    assert "the file is writable by group or others (mode 0620)" in warning

    allow_file.chmod(0o600)
    assert is_safe_upstream_url("http://127.0.0.1:4100/v1") is True


@_POSIX_USER
def test_a_file_in_a_group_writable_directory_is_refused_until_it_is_fixed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING, logger="headroom.proxy.upstream_guard")
    shared = tmp_path / "shared"
    shared.mkdir()
    allow_file = shared / "allowed"
    _replace(allow_file, "http://127.0.0.1:4100")
    monkeypatch.setenv(FILE_ENV, str(allow_file))
    shared.chmod(0o770)
    try:
        assert is_safe_upstream_url("http://127.0.0.1:4100/v1") is False
        assert is_safe_upstream_url("http://127.0.0.1:4100/v1") is False
    finally:
        # Fixing the directory changes nothing about the file, so this proves the failure was not cached against it.
        shared.chmod(0o700)

    assert is_safe_upstream_url("http://127.0.0.1:4100/v1") is True
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "its directory is writable by group or others (mode 0770)" in warnings[0]


@_POSIX_USER
def test_a_file_owned_by_another_user_is_refused(
    allow_file: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Ownership cannot be given away without root, so the running user is changed instead."""
    caplog.set_level(logging.WARNING, logger="headroom.proxy.upstream_guard")
    _replace(allow_file, "http://127.0.0.1:4100")
    real_uid = os.geteuid()
    monkeypatch.setattr(os, "geteuid", lambda: real_uid + 1)

    assert is_safe_upstream_url("http://127.0.0.1:4100/v1") is False
    (warning,) = [r.getMessage() for r in caplog.records]
    assert f"the file is owned by uid {real_uid}" in warning


def test_concurrent_checks_after_a_change_read_and_log_it_once(
    allow_file: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="headroom.proxy.upstream_guard")
    _replace(allow_file, "http://127.0.0.1:4100")
    assert is_safe_upstream_url("http://127.0.0.1:4100/v1") is True
    _replace(allow_file, "http://127.0.0.1:4200")
    caplog.clear()

    threads_count = 8
    arrived = threading.Barrier(threads_count)
    reads: list[str] = []
    real_read = upstream_guard._read_allowlist_file

    def slow_read(path: str, st: os.stat_result):  # noqa: ANN202
        reads.append(path)
        # Hold the reload long enough for every other check to stat the changed file and queue.
        threading.Event().wait(0.2)
        return real_read(path, st)

    monkeypatch.setattr(upstream_guard, "_read_allowlist_file", slow_read)
    results: list[bool] = []

    def check() -> None:
        arrived.wait(timeout=5)
        results.append(is_safe_upstream_url("http://127.0.0.1:4200/v1"))

    workers = [threading.Thread(target=check) for _ in range(threads_count)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=10)

    assert results == [True] * threads_count
    assert len(reads) == 1
    assert len([r for r in caplog.records if "loaded" in r.getMessage()]) == 1


@pytest.mark.parametrize(
    "entry",
    [
        "http://gw.example:8080 # prod",
        "http://gw.example:8080/ # prod",
        "http://gw.example:8080\t#prod",
    ],
)
def test_an_inline_comment_in_the_file_ends_the_entry(allow_file: Path, entry: str) -> None:
    _replace(allow_file, entry + "\n")

    assert is_safe_upstream_url("http://gw.example:8080/v1") is True


def test_a_comment_runs_to_the_end_of_its_line_only(allow_file: Path) -> None:
    allow_file.write_bytes(b"# first, http://127.0.0.1:4200\rhttp://127.0.0.1:4100 # a, b\n")

    assert is_safe_upstream_url("http://127.0.0.1:4100/v1") is True
    assert is_safe_upstream_url("http://127.0.0.1:4200/v1") is False


def test_a_hash_after_a_comma_is_an_entry_not_a_comment(
    allow_file: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="headroom.proxy.upstream_guard")
    _replace(allow_file, "http://127.0.0.1:4100,#http://127.0.0.1:4200\n")

    assert is_safe_upstream_url("http://127.0.0.1:4100/v1") is True
    assert is_safe_upstream_url("http://127.0.0.1:4200/v1") is False
    assert any("1 invalid entry skipped" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize(
    "entry", ["gw.example # prod", "a.example b.example", "http://gw.example:8080 # prod"]
)
def test_whitespace_inside_a_variable_entry_makes_it_invalid(
    monkeypatch: pytest.MonkeyPatch, entry: str, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING, logger="headroom.proxy.upstream_guard")
    monkeypatch.setenv(LIST_ENV, f"{entry},gateway.internal")

    assert is_safe_upstream_url("https://gateway.internal/v1") is True
    assert is_safe_upstream_url("https://gw.example/v1") is False
    assert is_safe_upstream_url("http://gw.example:8080/v1") is False
    assert [r.getMessage() for r in caplog.records] == [f"{LIST_ENV}: skipped 1 invalid entry"]


def test_whitespace_inside_a_file_entry_makes_it_invalid(
    allow_file: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="headroom.proxy.upstream_guard")
    _replace(allow_file, "a.example b.example\nhttp://127.0.0.1:4100\n")

    assert is_safe_upstream_url("https://a.example/v1") is False
    assert is_safe_upstream_url("http://127.0.0.1:4100/v1") is True
    assert any("1 invalid entry skipped" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize(
    "blank",
    ["\u00a0", "\x0c", "\u2028", "\u2029", "\x85", "\u3000"],
    ids=["nbsp", "form-feed", "line-separator", "paragraph-separator", "nel", "ideographic-space"],
)
def test_any_whitespace_before_a_hash_starts_a_comment(allow_file: Path, blank: str) -> None:
    """Text spaced off before a ``#`` is a comment, never a hidden entry after its comma."""
    _replace(allow_file, f"http://a.example{blank}# was, http://c.example\n")

    assert is_safe_upstream_url("http://a.example/v1") is True
    assert is_safe_upstream_url("http://c.example/v1") is False


@pytest.mark.parametrize(
    "brk", ["\x0b", "\x0c", "\x1c", "\x1d", "\x1e", "\x85", "\u2028", "\u2029"]
)
def test_every_line_break_ends_a_comment_and_separates_entries(allow_file: Path, brk: str) -> None:
    _replace(allow_file, f"# note{brk}http://127.0.0.1:4100{brk}http://127.0.0.1:4200\n")

    assert is_safe_upstream_url("http://127.0.0.1:4100/v1") is True
    assert is_safe_upstream_url("http://127.0.0.1:4200/v1") is True


def test_the_line_breaks_are_exactly_those_splitlines_recognises() -> None:
    boundaries = {chr(code) for code in range(0x110000) if len(f"a{chr(code)}b".splitlines()) == 2}
    assert set(upstream_guard._LINE_BREAKS) == boundaries


def test_the_comment_look_behind_uses_the_same_whitespace_as_entries() -> None:
    """``\\s`` must be ``str.isspace``, or the comment and the invalid-entry rules disagree."""
    import re

    pattern = re.compile(r"\s")
    disagree = [
        hex(code)
        for code in range(0x110000)
        if bool(pattern.match(chr(code))) != chr(code).isspace()
    ]
    assert disagree == []


@pytest.mark.parametrize(
    "bare",
    [
        "127.0.0.1:",
        "[::1]:",
        "gw.internal/v1",
        "gw.internal/",
        "user@gw.internal",
        "user:pw@gw.internal",
        "gw.internal?x=1",
        "gw.internal#frag",
    ],
)
def test_a_bare_entry_with_anything_but_a_hostname_is_invalid(
    allow_file: Path, monkeypatch: pytest.MonkeyPatch, bare: str, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="headroom.proxy.upstream_guard")
    _replace(allow_file, bare + "\n")
    monkeypatch.setenv(LIST_ENV, bare)

    for host in ("127.0.0.1", "[::1]", "gw.internal"):
        assert is_safe_upstream_url(f"http://{host}:4100/v1") is False
    assert any("1 invalid entry skipped" in r.getMessage() for r in caplog.records)
    assert any(r.getMessage() == f"{LIST_ENV}: skipped 1 invalid entry" for r in caplog.records)


@pytest.mark.parametrize("bare", ["gw.internal", "GW.Internal", "127.0.0.1", "[::1]"])
def test_a_bare_hostname_still_admits_every_port(allow_file: Path, bare: str) -> None:
    _replace(allow_file, bare + "\n")
    host = bare.lower()

    assert is_safe_upstream_url(f"http://{host}:4100/v1") is True
    assert is_safe_upstream_url(f"https://{host}/v1") is True


def test_a_blank_file_variable_is_treated_as_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(FILE_ENV, "  ")

    assert is_safe_upstream_url("https://8.8.8.8/v1") is True


def test_the_proxied_list_does_not_read_the_file(
    allow_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import httpx

    _replace(allow_file, "http://127.0.0.1:4100")

    assert proxied_upstream_listed(httpx.URL("http://127.0.0.1:4100/v1")) is False


def test_concurrent_checks_during_replacement_see_a_consistent_list(allow_file: Path) -> None:
    """Every answer is a whole version of the file, never a torn or empty one."""
    ports = (4100, 4101)
    expected = {(frozenset(), frozenset({("http", "127.0.0.1", port)})) for port in ports}
    _replace(allow_file, f"http://127.0.0.1:{ports[0]}")
    seen: list[object] = []
    stop = threading.Event()

    def check() -> None:
        while not stop.is_set():
            allow = upstream_guard.allowlisted_destinations()
            if allow not in expected:
                seen.append(allow)

    threads = [threading.Thread(target=check) for _ in range(4)]
    for thread in threads:
        thread.start()
    for n in range(200):
        _replace(allow_file, f"http://127.0.0.1:{ports[n % 2]}")
    stop.set()
    for thread in threads:
        thread.join()

    assert seen == []


class _Upstream:
    """A loopback upstream that records which paths reached it."""

    def __init__(self) -> None:
        self.hits: list[str] = []
        hits = self.hits

        class _Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                hits.append(self.path)
                body = b'{"id":"msg_test","type":"message","role":"assistant","content":[]}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args: object) -> None:
                return

        self._server = HTTPServer(("127.0.0.1", 0), _Handler)
        self.port = int(self._server.server_address[1])
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def __enter__(self) -> _Upstream:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._server.shutdown()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_a_running_proxy_admits_an_upstream_once_the_file_lists_it(allow_file: Path) -> None:
    """The scenario the file exists for: one app instance, no restart between requests."""
    _replace(allow_file, "http://127.0.0.1:1")
    app = create_app(
        ProxyConfig(
            host="127.0.0.1",
            port=_free_port(),
            optimize=False,
            cache_enabled=False,
            rate_limit_enabled=False,
        )
    )
    request = {
        "model": "claude-test",
        "max_tokens": 1,
        "messages": [{"role": "user", "content": "hi"}],
    }

    with _Upstream() as upstream, TestClient(app) as client:
        headers = {"x-headroom-base-url": upstream.url, "x-api-key": "test"}
        refused = client.post("/v1/messages", headers=headers, json=request)
        assert refused.status_code == 400
        assert upstream.hits == []

        _replace(allow_file, upstream.url)
        admitted = client.post("/v1/messages", headers=headers, json=request)

    assert admitted.status_code == 200, admitted.text
    assert upstream.hits == ["/v1/messages"]


class TestCliOption:
    """``headroom proxy --allowed-base-urls-file`` hands the path to the guard."""

    def _invoke(self, args: list[str], env: dict[str, str | None]) -> tuple[Result, str | None]:
        seen: dict[str, str | None] = {}

        def fake_run_server(config: object, **kwargs: object) -> None:
            seen["file"] = os.environ.get(FILE_ENV)

        with patch("headroom.proxy.server.run_server", fake_run_server):
            result = CliRunner().invoke(
                main,
                ["proxy", "--port", str(_free_port()), *args],
                env={"HEADROOM_HOST": None, "HEADROOM_PORT": None, **env},
            )
        return result, seen.get("file")

    def test_the_option_reaches_the_guard_before_the_server_starts(self, tmp_path: Path) -> None:
        path = tmp_path / "allowed-base-urls"

        result, seen = self._invoke(["--allowed-base-urls-file", str(path)], {FILE_ENV: None})

        assert result.exit_code == 0, result.output
        assert seen == str(path)

    def test_a_relative_path_is_made_absolute(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        upstream_guard.resolve_allowlist_file_path.cache_clear()

        result, seen = self._invoke(["--allowed-base-urls-file", "allowed"], {FILE_ENV: None})

        assert result.exit_code == 0, result.output
        assert seen == os.path.join(os.getcwd(), "allowed")

    def test_the_environment_variable_is_the_option_s_default(self, tmp_path: Path) -> None:
        path = tmp_path / "from-env"

        result, seen = self._invoke([], {FILE_ENV: str(path)})

        assert result.exit_code == 0, result.output
        assert seen == str(path)

    def test_a_directory_is_refused(self, tmp_path: Path) -> None:
        result, seen = self._invoke(["--allowed-base-urls-file", str(tmp_path)], {FILE_ENV: None})

        assert result.exit_code == 2
        assert "is a directory" in result.output
        assert seen is None

    @pytest.mark.parametrize(
        ("args", "env"),
        [
            (["--allowed-base-urls-file", ""], {FILE_ENV: None}),
            (["--allowed-base-urls-file", "   "], {FILE_ENV: None}),
            ([], {FILE_ENV: "   "}),
            (["--allowed-base-urls-file", ""], {FILE_ENV: "/somewhere/else"}),
        ],
        ids=["blank-option", "spaces-option", "spaces-env", "blank-option-overrides-env"],
    )
    def test_a_blank_value_means_no_file(self, args: list[str], env: dict[str, str | None]) -> None:
        result, seen = self._invoke(args, env)

        assert result.exit_code == 0, result.output
        assert seen is None

    def test_a_home_relative_path_is_expanded(self, tmp_path: Path) -> None:
        upstream_guard.resolve_allowlist_file_path.cache_clear()
        result, seen = self._invoke(
            ["--allowed-base-urls-file", "~/allowed"], {FILE_ENV: None, "HOME": str(tmp_path)}
        )

        assert result.exit_code == 0, result.output
        assert seen == str(tmp_path / "allowed")

    def test_the_option_is_in_the_help(self) -> None:
        result = CliRunner().invoke(main, ["proxy", "--help"])

        assert "--allowed-base-urls-file" in result.output
        assert "HEADROOM_ALLOWED_BASE_URLS_FILE" in result.output
