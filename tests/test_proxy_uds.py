"""Tests for serving the proxy on a Unix domain socket (`headroom proxy --uds`).

The socket transport is a plain alternative to a TCP port; see
`headroom/proxy/unix_socket.py` for the socket's mode, its directory rule and
stale-socket handling, which `tests/test_proxy/test_unix_socket.py` covers, and
`docs/content/docs/troubleshooting.mdx` for why it does not restore Claude
Code's Remote Control (GH #1779).
"""

from __future__ import annotations

import socket
import stat
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from headroom.cli.proxy import proxy as proxy_cmd
from headroom.proxy.unix_socket import (
    SOCKET_MODE,
    UnixSocketUnusableError,
    require_unix_sockets,
    socket_usage_lines,
)

requires_uds = pytest.mark.skipif(
    not hasattr(socket, "AF_UNIX"), reason="platform has no socket.AF_UNIX (Windows)"
)

try:  # `headroom.proxy.server` pulls in the compiled Rust core.
    import headroom._core  # noqa: F401

    _CORE_BUILT = True
except ImportError:  # pragma: no cover - depends on the local build
    _CORE_BUILT = False

requires_core = pytest.mark.skipif(
    not _CORE_BUILT, reason="headroom._core is not built in this environment"
)


@pytest.fixture
def socket_dir() -> Iterator[Path]:
    # mkdtemp creates the directory with mode 0700, the private parent --uds requires. Not pytest's tmp_path, because AF_UNIX paths are limited to about a hundred bytes.
    path = Path(tempfile.mkdtemp(prefix="hr-uds-"))
    yield path
    for child in path.iterdir():
        child.unlink()
    path.rmdir()


# --------------------------------------------------------------------------
# Platform capability
# --------------------------------------------------------------------------


def test_require_unix_sockets_rejects_windows_even_with_af_unix() -> None:
    """asyncio has no Windows unix socket transport, whatever the socket module defines."""
    with (
        patch.object(sys, "platform", "win32"),
        pytest.raises(UnixSocketUnusableError, match="win32 does not provide"),
    ):
        require_unix_sockets()


@requires_uds
def test_require_unix_sockets_accepts_posix() -> None:
    require_unix_sockets()


def test_cli_rejects_uds_on_windows() -> None:
    """The CLI fails fast with a readable error, not a bind-time OSError."""
    with patch.object(sys, "platform", "win32"):
        result = CliRunner().invoke(proxy_cmd, ["--uds", "/tmp/headroom-test.sock"])

    assert result.exit_code != 0
    assert "unix domain sockets" in result.output
    assert "--port instead" in result.output


# --------------------------------------------------------------------------
# Startup banner: a socket bind must not advertise a broken recipe.
# --------------------------------------------------------------------------


def test_socket_usage_lines_omit_the_unsupported_claude_code_recipe() -> None:
    """Regression: the banner once printed a configuration that cannot work.

    `ANTHROPIC_UNIX_SOCKET=... claude` passes Claude Code's api.anthropic.com
    host check but reclassifies the session as API-key auth, and the session
    then fails to authenticate. Printing it at startup turned a known-negative
    field result into first-party runtime guidance.
    """
    rendered = "\n".join(socket_usage_lines("/run/headroom/proxy.sock"))

    assert "ANTHROPIC_UNIX_SOCKET" not in rendered
    assert "ANTHROPIC_BASE_URL" not in rendered
    assert "claude" not in rendered.lower()


def test_socket_usage_lines_state_the_transport_requirement() -> None:
    """What replaces the recipe has to be useful, not merely absent."""
    path = "/run/headroom/proxy.sock"

    rendered = "\n".join(socket_usage_lines(path))

    assert path in rendered
    assert "HTTP over a Unix socket" in rendered
    assert "curl --unix-socket" in rendered
    assert "serving-on-a-unix-socket" in rendered


def test_socket_usage_lines_name_no_agent() -> None:
    """Transport-neutral: the banner singles out no client."""
    rendered = "\n".join(socket_usage_lines("/run/headroom/proxy.sock")).lower()

    for agent in ("claude", "codex", "opencode", "cursor", "aider", "copilot"):
        assert agent not in rendered, f"banner should not name {agent}"


# --------------------------------------------------------------------------
# Server wiring: uvicorn is mocked.
# --------------------------------------------------------------------------


def _bind_kwargs_for(**config_kwargs: object) -> dict[str, object]:
    """Run run_server far enough to capture what it would bind to, and the socket's mode then."""
    from headroom.proxy.server import ProxyConfig, run_server

    captured: dict[str, object] = {}

    def fake_run_uvicorn(  # noqa: ANN202
        app_target,  # noqa: ANN001
        bind_kwargs,  # noqa: ANN001
        workers,  # noqa: ANN001
        limit_concurrency,  # noqa: ANN001
        log_level,  # noqa: ANN001
        uvicorn_kwargs,  # noqa: ANN001
    ):
        captured.update(bind_kwargs)
        if "fd" in bind_kwargs:
            sock = socket.socket(fileno=bind_kwargs["fd"])
            try:
                captured["path"] = sock.getsockname()
                captured["mode"] = stat.S_IMODE(Path(sock.getsockname()).lstat().st_mode)
            finally:
                sock.detach()

    with (
        patch("headroom.proxy.server._run_uvicorn", side_effect=fake_run_uvicorn),
        patch("headroom.proxy.server.create_app"),
    ):
        run_server(ProxyConfig(**config_kwargs), print_banner=False)  # type: ignore[arg-type]

    return captured


@requires_core
def test_run_server_binds_host_and_port_by_default() -> None:
    bind = _bind_kwargs_for(host="127.0.0.1", port=9123)

    assert bind == {"host": "127.0.0.1", "port": 9123}


@requires_uds
@requires_core
def test_run_server_hands_uvicorn_an_owner_only_socket_instead_of_a_port(
    socket_dir: Path,
) -> None:
    """uvicorn gets an already bound descriptor, never a path it would chmod world-writable."""
    target = socket_dir / "headroom.sock"

    bind = _bind_kwargs_for(uds=str(target))

    assert set(bind) == {"fd", "path", "mode"}
    assert bind["path"] == str(target)
    assert bind["mode"] == SOCKET_MODE


@requires_uds
@requires_core
def test_run_server_removes_the_socket_on_exit(socket_dir: Path) -> None:
    """A crash inside uvicorn must not leave an inode that blocks the restart."""
    from headroom.proxy.server import ProxyConfig, run_server

    target = socket_dir / "headroom.sock"

    def fail(*_args: object) -> None:
        assert target.is_socket()
        raise KeyboardInterrupt

    with (
        patch("headroom.proxy.server._run_uvicorn", side_effect=fail),
        patch("headroom.proxy.server.create_app"),
        pytest.raises(KeyboardInterrupt),
    ):
        run_server(ProxyConfig(uds=str(target)), print_banner=False)

    assert not target.exists()
