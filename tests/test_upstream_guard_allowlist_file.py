"""Tests for HEADROOM_ALLOWED_BASE_URLS_FILE, the allowlist the guard re-reads at runtime.

A supervisor that lists its own loopback endpoint cannot restart a proxy that sessions are still using when that endpoint's port changes, so the guard has to pick up a rewritten file on the next request. Every case uses IP literals, so no external network is needed.
"""

from __future__ import annotations

import logging
import os
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

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
    reads: list[str] = []
    real_open = open

    def counting_open(file: object, *args: object, **kwargs: object):  # noqa: ANN202
        if str(file) == str(allow_file):
            reads.append(str(file))
        return real_open(file, *args, **kwargs)  # type: ignore[call-overload]

    monkeypatch.setattr(upstream_guard, "open", counting_open, raising=False)

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
