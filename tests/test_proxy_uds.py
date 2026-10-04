"""Tests for serving the proxy on a Unix domain socket (`headroom proxy --uds`).

The socket transport is a plain alternative to a TCP port — see
`headroom/proxy/uds.py` for the rationale and for why it does not restore
Claude Code's Remote Control (GH #1779).
"""

from __future__ import annotations

import os
import shutil
import socket
import stat
import tempfile
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from headroom.cli.proxy import proxy as proxy_cmd
from headroom.proxy.uds import (
    UDS_SUPPORTED,
    UdsError,
    _missing_ancestors,
    _require_safe_existing_parent,
    bind_uds_listener,
    max_uds_path_length,
    prepare_uds_path,
    require_uds_support,
    socket_usage_lines,
)

requires_uds = pytest.mark.skipif(
    not UDS_SUPPORTED, reason="platform has no socket.AF_UNIX (Windows)"
)

# The POSIX temp root. pytest's tmp_path lives under $TMPDIR, which on macOS is a per-user /var/folders/... path deep enough that a socket nested a few levels below it overruns the 104-byte sun_path limit, so the bind-level tests would fail for a reason unrelated to the code under test.
_SHORT_TEMP_ROOT = "/tmp"


@pytest.fixture
def sock_dir() -> Iterator[Path]:
    """A fresh private directory whose paths fit in sun_path on every POSIX platform."""
    path = Path(tempfile.mkdtemp(prefix="hr-uds-", dir=_SHORT_TEMP_ROOT))
    yield path
    shutil.rmtree(path)


try:  # `headroom.proxy.server` pulls in the compiled Rust core.
    import headroom._core  # noqa: F401

    _CORE_BUILT = True
except ImportError:  # pragma: no cover - depends on the local build
    _CORE_BUILT = False

requires_core = pytest.mark.skipif(
    not _CORE_BUILT, reason="headroom._core is not built in this environment"
)


# --------------------------------------------------------------------------
# Platform capability — runs everywhere, since the platform is a parameter.
# --------------------------------------------------------------------------


def test_require_uds_support_rejects_windows() -> None:
    """Windows has neither socket.AF_UNIX nor an asyncio UDS transport."""
    with pytest.raises(UdsError, match="unavailable on this platform"):
        require_uds_support(platform="win32")


def test_require_uds_support_accepts_posix() -> None:
    if not UDS_SUPPORTED:
        pytest.skip("AF_UNIX missing; the platform argument cannot override that")
    require_uds_support(platform="linux")


def test_sun_path_limit_is_platform_specific() -> None:
    """Linux allows 108 bytes, the BSDs and macOS 104. Guessing high truncates."""
    assert max_uds_path_length("linux") == 108
    assert max_uds_path_length("darwin") == 104


def test_cli_rejects_uds_on_windows() -> None:
    """The CLI fails fast with a readable error, not a bind-time OSError."""
    with patch("headroom.proxy.uds.UDS_SUPPORTED", False):
        result = CliRunner().invoke(proxy_cmd, ["--uds", "/tmp/headroom-test.sock"])

    assert result.exit_code != 0
    assert "Unix domain sockets" in result.output
    assert "--port instead" in result.output


# --------------------------------------------------------------------------
# Parent-directory policy — pure logic, so it runs on every platform.
# --------------------------------------------------------------------------


def test_missing_ancestors_lists_only_absent_levels(sock_dir: Path) -> None:
    """Only these get chmod 0700; anything already on disk is left alone."""
    existing = sock_dir / "existing"
    existing.mkdir()

    missing = _missing_ancestors(existing / "a" / "b")

    assert missing == [existing / "a", existing / "a" / "b"]


def test_missing_ancestors_is_empty_for_an_existing_dir(sock_dir: Path) -> None:
    assert _missing_ancestors(sock_dir) == []


class _FakeStat:
    def __init__(self, mode: int, uid: int | None = None) -> None:
        self.st_mode = mode
        self.st_uid = os.geteuid() if uid is None else uid


@pytest.mark.parametrize(
    ("mode", "accepted"),
    [
        (0o700, True),  # owner only
        (0o750, True),  # group may read/traverse, not write
        (0o755, True),  # the common shared-parent case
        (0o770, False),  # any group member could swap the socket
        (0o777, False),  # any local user could
        (0o1777, True),  # /tmp: sticky, so others cannot unlink ours
        (0o1770, True),  # sticky group-writable
    ],
)
@requires_uds
def test_existing_parent_accepted_only_when_others_cannot_swap_the_socket(
    sock_dir: Path, mode: int, accepted: bool
) -> None:
    """The mode is injected rather than applied, so every case runs without chmod."""
    with patch.object(Path, "stat", return_value=_FakeStat(stat.S_IFDIR | mode)):
        if accepted:
            _require_safe_existing_parent(sock_dir)
        else:
            with pytest.raises(UdsError, match="writable by other users"):
                _require_safe_existing_parent(sock_dir)


@requires_uds
@pytest.mark.parametrize(
    ("owner", "accepted"),
    [("self", True), ("root", True), ("other", False)],
)
def test_existing_parent_must_be_owned_by_us_or_root(
    sock_dir: Path, owner: str, accepted: bool
) -> None:
    """A directory's owner can unlink any entry in it, so another user's 0700 directory is not private to us."""
    uid = {"self": os.geteuid(), "root": 0, "other": os.geteuid() + 1}[owner]
    with patch.object(Path, "stat", return_value=_FakeStat(stat.S_IFDIR | 0o700, uid)):
        if accepted:
            _require_safe_existing_parent(sock_dir)
        else:
            with pytest.raises(UdsError, match="is owned by uid"):
                _require_safe_existing_parent(sock_dir)


def test_unreadable_existing_parent_defers_to_bind(sock_dir: Path) -> None:
    """A stat we cannot perform is not evidence of a problem; let bind() rule."""
    with patch.object(Path, "stat", side_effect=PermissionError):
        _require_safe_existing_parent(sock_dir)


# --------------------------------------------------------------------------
# Startup banner — a socket bind must not advertise a broken recipe.
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
# Path preparation — needs a real AF_UNIX platform.
# --------------------------------------------------------------------------


@requires_uds
def test_prepare_creates_parent_owner_only(sock_dir: Path) -> None:
    """The directory mode is the access-control boundary for the socket."""
    target = sock_dir / "run" / "headroom.sock"

    resolved = prepare_uds_path(target)

    assert resolved == target
    assert target.parent.is_dir()
    assert stat.S_IMODE(target.parent.stat().st_mode) == 0o700


@requires_uds
def test_prepare_preserves_an_existing_parents_mode(sock_dir: Path) -> None:
    """A caller-owned directory must not be silently tightened to 0700.

    Regression test: `--uds /run/shared/hr.sock` where `/run/shared` is a
    directory someone else set up at 0755 would have locked out every other
    user of that directory.
    """
    parent = sock_dir / "shared"
    parent.mkdir()
    parent.chmod(0o755)
    bystander = parent / "someone-elses.txt"
    bystander.write_text("theirs", encoding="utf-8")

    prepare_uds_path(parent / "headroom.sock")

    assert stat.S_IMODE(parent.stat().st_mode) == 0o755
    assert bystander.read_text(encoding="utf-8") == "theirs"


@requires_uds
def test_prepare_only_chmods_directories_it_creates(sock_dir: Path) -> None:
    """The 0700 applies to the new levels, not to the existing root above them."""
    root = sock_dir / "existing"
    root.mkdir()
    root.chmod(0o755)

    prepare_uds_path(root / "a" / "b" / "headroom.sock")

    assert stat.S_IMODE(root.stat().st_mode) == 0o755
    assert stat.S_IMODE((root / "a").stat().st_mode) == 0o700
    assert stat.S_IMODE((root / "a" / "b").stat().st_mode) == 0o700


@requires_uds
def test_prepare_refuses_a_world_writable_existing_parent(sock_dir: Path) -> None:
    """Without the sticky bit, any local user could swap the socket out."""
    parent = sock_dir / "open"
    parent.mkdir()
    parent.chmod(0o777)

    with pytest.raises(UdsError, match="writable by other users"):
        prepare_uds_path(parent / "headroom.sock")

    assert stat.S_IMODE(parent.stat().st_mode) == 0o777, "the refusal must not mutate"


@requires_uds
def test_prepare_accepts_a_sticky_world_writable_parent(sock_dir: Path) -> None:
    """`/tmp` is 1777: others can add entries but cannot unlink ours."""
    parent = sock_dir / "sticky"
    parent.mkdir()
    parent.chmod(0o1777)

    resolved = prepare_uds_path(parent / "headroom.sock")

    assert resolved.parent == parent
    assert stat.S_IMODE(parent.stat().st_mode) == 0o1777


@requires_uds
def test_prepare_clears_a_stale_socket(sock_dir: Path) -> None:
    """A crashed proxy leaves an inode behind; a restart must not trip on it."""
    target = sock_dir / "stale.sock"
    dead = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    dead.bind(str(target))
    dead.close()  # closing without unlinking is exactly the crash case
    assert target.exists()

    prepare_uds_path(target)

    assert not target.exists()


# A tiny ASGI app stands in for the real one so the child starts in well under this.
_SIGTERM_STARTUP_DEADLINE_SECS = 30.0

_SIGTERM_CHILD = """
import sys
from unittest.mock import patch

from headroom.proxy.server import ProxyConfig, run_server


async def app(scope, receive, send):
    if scope["type"] == "http":
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})


with patch("headroom.proxy.server.create_app", return_value=app):
    run_server(ProxyConfig(uds=sys.argv[1]), print_banner=False)
"""


@requires_uds
@requires_core
def test_sigterm_removes_the_socket_and_still_exits_by_signal(sock_dir: Path) -> None:
    """uvicorn re-raises SIGTERM after shutdown, which under the default handler skips every finally block."""
    import signal
    import subprocess
    import sys
    import time

    target = sock_dir / "headroom.sock"
    child = subprocess.Popen(  # noqa: S603
        [sys.executable, "-c", _SIGTERM_CHILD, str(target)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    try:
        deadline = time.monotonic() + _SIGTERM_STARTUP_DEADLINE_SECS
        while True:
            assert child.poll() is None, child.stderr.read().decode() if child.stderr else ""
            probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                probe.connect(str(target))
                break
            except OSError:
                assert time.monotonic() < deadline, "the proxy never started listening"
                time.sleep(0.1)
            finally:
                probe.close()

        child.send_signal(signal.SIGTERM)
        returncode = child.wait(timeout=_SIGTERM_STARTUP_DEADLINE_SECS)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()

    assert returncode == -signal.SIGTERM
    assert not target.exists(), "the socket file must not outlive the proxy"


@requires_uds
def test_prepare_refuses_a_live_socket(sock_dir: Path) -> None:
    """Two proxies on one socket would silently steal each other's traffic."""
    target = sock_dir / "live.sock"
    live = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    live.bind(str(target))
    live.listen(1)
    try:
        with pytest.raises(UdsError, match="already listening"):
            prepare_uds_path(target)
        assert target.exists(), "the live socket must survive the refusal"
    finally:
        live.close()
        target.unlink(missing_ok=True)


@requires_uds
def test_prepare_never_deletes_a_regular_file(sock_dir: Path) -> None:
    """A typo'd --uds pointing at real data must not destroy it."""
    target = sock_dir / "notes.txt"
    target.write_text("important", encoding="utf-8")

    with pytest.raises(UdsError, match="is not a socket"):
        prepare_uds_path(target)

    assert target.read_text(encoding="utf-8") == "important"


@requires_uds
def test_prepare_rejects_an_oversized_path(sock_dir: Path) -> None:
    """Past sun_path, bind() fails with an ENAMETOOLONG that names nothing."""
    target = sock_dir / ("d" * 120) / "headroom.sock"

    with pytest.raises(UdsError, match="sun_path limit"):
        prepare_uds_path(target)


@requires_uds
def test_oversized_path_advice_does_not_point_at_tmpdir(sock_dir: Path) -> None:
    """On macOS $TMPDIR is a deep per-user /var/folders path, so advising it sends users somewhere that fails the same check."""
    target = sock_dir / ("d" * 120) / "headroom.sock"

    with pytest.raises(UdsError) as excinfo:
        prepare_uds_path(target)

    assert "under $TMPDIR" not in str(excinfo.value)
    assert "/tmp" in str(excinfo.value)


@requires_uds
@pytest.mark.parametrize("parent_mode", [0o755, 0o1777])
def test_bound_socket_is_owner_only_in_a_shared_parent(sock_dir: Path, parent_mode: int) -> None:
    """An existing 0755 or sticky parent is accepted, so the socket's own mode must keep other users out."""
    parent = sock_dir / "shared"
    parent.mkdir()
    parent.chmod(parent_mode)

    listener = bind_uds_listener(prepare_uds_path(parent / "headroom.sock"))
    try:
        assert stat.S_IMODE((parent / "headroom.sock").stat().st_mode) == 0o600
    finally:
        listener.close()

    assert not (parent / "headroom.sock").exists()


@requires_uds
def test_close_leaves_a_successors_socket_alone(sock_dir: Path) -> None:
    """Once this proxy stops accepting, a successor may replace its socket; shutdown must not unlink that one."""
    target = sock_dir / "headroom.sock"
    listener = bind_uds_listener(prepare_uds_path(target))
    listener.sock.close()  # stopped accepting: the file now looks stale
    successor = bind_uds_listener(prepare_uds_path(target))
    try:
        listener.close()

        assert target.is_socket(), "the successor's socket must survive"
    finally:
        successor.close()

    assert not target.exists()


@requires_uds
def test_close_tolerates_an_already_removed_file(sock_dir: Path) -> None:
    target = sock_dir / "headroom.sock"
    listener = bind_uds_listener(prepare_uds_path(target))
    target.unlink()

    listener.close()

    assert not target.exists()


# --------------------------------------------------------------------------
# Server wiring — uvicorn is mocked, so this runs on every platform.
# --------------------------------------------------------------------------


def _bind_kwargs_for(**config_kwargs: object) -> dict[str, object]:
    """Run run_server far enough to capture what it would bind to."""
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
            bound = socket.socket(fileno=os.dup(bind_kwargs["fd"]))
            try:
                captured["socket_mode"] = stat.S_IMODE(os.stat(bound.getsockname()).st_mode)
            finally:
                bound.close()

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
def test_run_server_binds_the_socket_instead_of_a_port(sock_dir: Path) -> None:
    """uvicorn treats uds and host/port as alternatives; passing both is an error."""
    target = sock_dir / "headroom.sock"

    bind = _bind_kwargs_for(host="127.0.0.1", port=9123, uds=str(target))

    assert set(bind) == {"fd", "socket_mode"}
    assert bind["socket_mode"] == 0o600, "uvicorn must receive a socket already narrowed to 0600"


@requires_uds
@requires_core
def test_run_server_removes_the_socket_on_exit(sock_dir: Path) -> None:
    """A crash inside uvicorn must not leave an inode that blocks the restart."""
    from headroom.proxy.server import ProxyConfig, run_server

    target = sock_dir / "headroom.sock"

    def bind_then_fail(  # noqa: ANN202
        app_target,  # noqa: ANN001
        bind_kwargs,  # noqa: ANN001
        workers,  # noqa: ANN001
        limit_concurrency,  # noqa: ANN001
        log_level,  # noqa: ANN001
        uvicorn_kwargs,  # noqa: ANN001
    ):
        raise KeyboardInterrupt

    with (
        patch("headroom.proxy.server._run_uvicorn", side_effect=bind_then_fail),
        patch("headroom.proxy.server.create_app"),
        pytest.raises(KeyboardInterrupt),
    ):
        run_server(ProxyConfig(uds=str(target)), print_banner=False)

    assert not target.exists()


# --------------------------------------------------------------------------
# Per-instance state: a socket proxy is not identified by the port it ignores.
# --------------------------------------------------------------------------


@requires_core
def test_instance_key_separates_socket_proxies_from_each_other_and_from_tcp() -> None:
    from headroom.proxy.models import ProxyConfig

    tcp = ProxyConfig(port=8787)
    first = ProxyConfig(port=8787, uds="/tmp/hr-a/proxy.sock")
    second = ProxyConfig(port=8787, uds="/tmp/hr-b/proxy.sock")

    assert tcp.instance_key == 8787
    keys = {tcp.instance_key, first.instance_key, second.instance_key}
    assert len(keys) == 3
    assert str(first.instance_key).startswith("uds-")


@requires_core
def test_socket_proxy_state_is_not_named_after_the_ignored_port(sock_dir: Path) -> None:
    """`--uds X --port 8798` must not name its beacon lock or sidecar socket after 8798."""
    from headroom import paths
    from headroom.cli.proxy import default_embedding_socket
    from headroom.proxy.models import ProxyConfig

    config = ProxyConfig(port=8798, uds=str(sock_dir / "proxy.sock"))

    assert "8798" not in paths.beacon_lock_path(config.instance_key).name
    assert "8798" not in default_embedding_socket(config)
    assert len(default_embedding_socket(config).encode()) < max_uds_path_length()


@requires_uds
@requires_core
def test_relative_and_absolute_spellings_share_one_instance_key(
    sock_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from headroom.proxy.models import ProxyConfig

    monkeypatch.chdir(sock_dir)

    relative = ProxyConfig(uds="proxy.sock").instance_key
    absolute = ProxyConfig(uds=str(Path.cwd() / "proxy.sock")).instance_key

    assert relative == absolute


# --------------------------------------------------------------------------
# Listen-address conflicts: a socket path never silently overrides a port.
# --------------------------------------------------------------------------


@requires_uds
@pytest.mark.parametrize(
    ("args", "env", "named"),
    [
        (["--uds", "/tmp/hr.sock", "--port", "8798"], {}, "--port"),
        (["--uds", "/tmp/hr.sock", "--host", "127.0.0.1"], {}, "--host"),
        (["--uds", "/tmp/hr.sock"], {"HEADROOM_PORT": "8798"}, "--port"),
        (["--port", "8798"], {"HEADROOM_UDS": "/tmp/hr.sock"}, "--port"),
    ],
)
def test_uds_with_an_explicit_tcp_address_is_a_usage_error(
    args: list[str], env: dict[str, str], named: str
) -> None:
    result = CliRunner().invoke(proxy_cmd, args, env=env)

    assert result.exit_code == 2, result.output
    assert "cannot be combined with" in result.output
    assert named in result.output


def test_headroom_uds_is_not_claimed_by_the_install_manifest() -> None:
    """No install manifest sets HEADROOM_UDS, so marking it manifest-managed misdescribed it."""
    from headroom.settings_store import SETTINGS

    (field,) = [f for f in SETTINGS if f.env == "HEADROOM_UDS"]
    assert not field.manifest_managed
    assert "manifest" not in field.help


# --------------------------------------------------------------------------
# cc-switch reconciler: it writes a TCP URL, so it cannot serve a socket proxy.
# --------------------------------------------------------------------------


@requires_uds
def test_cli_refuses_the_cc_switch_reconciler_with_uds() -> None:
    result = CliRunner().invoke(
        proxy_cmd, ["--uds", "/tmp/hr.sock"], env={"HEADROOM_CC_SWITCH_RECONCILE": "1"}
    )

    assert result.exit_code == 1, result.output
    assert "HEADROOM_CC_SWITCH_RECONCILE" in result.output
    assert "Traceback" not in result.output


def test_reconciler_refusal_applies_only_to_socket_listeners(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """create_app calls this too, so `python -m headroom.proxy.server` is covered."""
    from headroom.proxy.cc_switch_reconciler import refuse_unix_socket_listener

    monkeypatch.setenv("HEADROOM_CC_SWITCH_RECONCILE", "1")
    refuse_unix_socket_listener(None)
    with pytest.raises(ValueError, match="does not listen on"):
        refuse_unix_socket_listener("/tmp/hr.sock")

    monkeypatch.delenv("HEADROOM_CC_SWITCH_RECONCILE")
    refuse_unix_socket_listener("/tmp/hr.sock")


# --------------------------------------------------------------------------
# Trust model: a socket peer is local for admin routes but cannot pick a partition.
# --------------------------------------------------------------------------


def test_socket_caller_cannot_select_a_memory_partition(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pins the documented trust model: no peer address means no header trust."""
    from types import SimpleNamespace

    from headroom.proxy.identity import USER_ID_HEADER, resolve_memory_identity

    monkeypatch.delenv("HEADROOM_PROXY_TOKEN", raising=False)
    socket_request = SimpleNamespace(client=None, headers={USER_ID_HEADER: "someone-else"})

    assert resolve_memory_identity(socket_request, default="proxy-owner") == "proxy-owner"


def test_socket_caller_is_local_for_loopback_only_routes() -> None:
    """The /debug guard and the token exemption admit a peer with no address; the 0600 socket mode is what justifies that."""
    from headroom.proxy.loopback_guard import is_loopback_host

    assert is_loopback_host(None)
