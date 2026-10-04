"""Unix domain socket listener: ``headroom proxy --uds PATH`` and ``run_server`` with ``ProxyConfig.uds``."""

from __future__ import annotations

import os
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")
httpx = pytest.importorskip("httpx")

from click.testing import CliRunner  # noqa: E402

from headroom.cli.main import main  # noqa: E402
from headroom.proxy.models import ProxyConfig  # noqa: E402
from headroom.proxy.unix_socket import (  # noqa: E402
    SOCKET_MODE,
    UnixSocketInUseError,
    UnixSocketUnusableError,
    bind_unix_listener,
    max_socket_path_bytes,
    require_unix_sockets,
    resolve_unix_socket_path,
)

pytestmark = pytest.mark.skipif(not hasattr(socket, "AF_UNIX"), reason="needs AF_UNIX")

REPO_ROOT = Path(__file__).resolve().parents[2]

# Runs the real run_server with create_app replaced by a small app, so the test exercises the listener path (binding, uvicorn hand-off, shutdown cleanup) without the proxy's startup cost.
_SERVER_SCRIPT = textwrap.dedent(
    """
    import os, socket, sys
    from fastapi import Depends, FastAPI, Request
    import headroom.proxy.server as server
    from headroom.proxy.loopback_guard import require_loopback
    from headroom.proxy.models import ProxyConfig

    app = FastAPI()

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/debug/peer", dependencies=[Depends(require_loopback)])
    def peer(request: Request):
        return {"client": request.client}

    @app.get("/sockets")
    def sockets():
        found = []
        for name in os.listdir("/dev/fd"):
            try:
                sock = socket.socket(fileno=os.dup(int(name)))
            except OSError:
                continue
            try:
                found.append([sock.family.name, str(sock.getsockname())])
            finally:
                sock.close()
        return {"sockets": found}

    server.create_app = lambda config: app
    server.run_server(ProxyConfig(uds=sys.argv[1]), print_banner=False)
    """
)


@pytest.fixture
def socket_dir() -> Path:
    # mkdtemp creates the directory with mode 0700, the private parent a caller must provide. It is not pytest's tmp_path because AF_UNIX paths are limited to about a hundred bytes.
    path = Path(tempfile.mkdtemp(prefix="hr-uds-"))
    yield path
    for child in path.iterdir():
        child.unlink()
    path.rmdir()


def _wait_for_socket(path: Path, proc: subprocess.Popen[str]) -> None:
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise AssertionError(f"server exited early: {proc.communicate()}")
        if path.exists():
            try:
                with httpx.Client(transport=httpx.HTTPTransport(uds=str(path))) as client:
                    client.get("http://localhost/health")
                return
            except httpx.TransportError:
                pass
        time.sleep(0.1)
    raise AssertionError("server never answered on the socket")


@pytest.fixture
def running_server(socket_dir: Path):
    path = socket_dir / "proxy.sock"
    proc = subprocess.Popen(
        [sys.executable, "-c", _SERVER_SCRIPT, str(path)],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _wait_for_socket(path, proc)
        yield path, proc
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            proc.wait(timeout=30)


def _get(path: Path, url_path: str) -> httpx.Response:
    with httpx.Client(transport=httpx.HTTPTransport(uds=str(path))) as client:
        return client.get(f"http://localhost{url_path}")


class TestServedOverUnixSocket:
    def test_answers_http_on_the_socket(self, running_server):
        path, _ = running_server
        response = _get(path, "/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_socket_is_owner_only(self, running_server):
        path, _ = running_server
        mode = path.stat().st_mode
        assert stat.S_ISSOCK(mode)
        assert stat.S_IMODE(mode) == SOCKET_MODE

    def test_binds_no_tcp_listener(self, running_server):
        path, _ = running_server
        sockets = _get(path, "/sockets").json()["sockets"]
        assert ["AF_UNIX", str(path)] in sockets
        assert [family for family, _ in sockets if family != "AF_UNIX"] == []

    def test_debug_guard_admits_socket_peer_without_client_address(self, running_server):
        path, _ = running_server
        response = _get(path, "/debug/peer")
        assert response.status_code == 200
        assert response.json() == {"client": None}

    def test_socket_removed_on_clean_shutdown(self, running_server):
        path, proc = running_server
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=30)
        # The process still ends by the signal, as uvicorn's own re-raise makes it do over TCP.
        assert proc.returncode == -signal.SIGTERM, proc.communicate()
        assert not path.exists()


class TestBindUnixListener:
    def test_replaces_stale_socket(self, socket_dir: Path):
        path = socket_dir / "proxy.sock"
        stale = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        stale.bind(str(path))
        stale.close()

        listener = bind_unix_listener(str(path))
        try:
            assert stat.S_IMODE(path.stat().st_mode) == SOCKET_MODE
            # The path now reaches the new socket: a client connects and the listener accepts it. Inode numbers cannot show this, because Linux reuses them after unlink.
            listener.sock.listen()
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                client.connect(str(path))
                accepted, _ = listener.sock.accept()
                accepted.close()
        finally:
            listener.close()
        assert not path.exists()

    def test_refuses_live_listener(self, socket_dir: Path):
        path = socket_dir / "proxy.sock"
        live = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        live.bind(str(path))
        live.listen()
        try:
            with pytest.raises(UnixSocketInUseError, match="already listening"):
                bind_unix_listener(str(path))
            assert path.exists()
        finally:
            live.close()

    def test_refuses_non_socket(self, socket_dir: Path):
        path = socket_dir / "proxy.sock"
        path.write_text("not a socket")
        with pytest.raises(UnixSocketInUseError, match="not a socket"):
            bind_unix_listener(str(path))
        assert path.read_text() == "not a socket"

    def test_close_leaves_a_successor_socket(self, socket_dir: Path):
        path = socket_dir / "proxy.sock"
        listener = bind_unix_listener(str(path))
        path.unlink()
        successor = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        successor.bind(str(path))
        try:
            listener.close()
            assert stat.S_ISSOCK(path.stat().st_mode)
        finally:
            successor.close()


class TestSocketPathResolution:
    def test_limit_is_the_longest_path_a_bind_accepts(self, socket_dir: Path):
        limit = max_socket_path_bytes()
        fitting = socket_dir / ("a" * (limit - len(str(socket_dir)) - 1))
        assert len(os.fsencode(fitting)) == limit
        listener = bind_unix_listener(str(fitting))
        listener.close()
        with pytest.raises(UnixSocketUnusableError, match=rf"is {limit + 1} bytes, longer than"):
            bind_unix_listener(f"{fitting}b")

    def test_error_names_resolved_length_and_limit(self, socket_dir: Path):
        path = str(socket_dir / ("a" * 300))
        length = len(path)
        with pytest.raises(UnixSocketUnusableError) as caught:
            resolve_unix_socket_path(path)
        message = str(caught.value)
        assert path in message
        assert f"{length} bytes" in message
        assert f"the {max_socket_path_bytes()} bytes" in message

    def test_relative_path_is_resolved_against_the_current_directory(
        self, socket_dir: Path, monkeypatch
    ):
        monkeypatch.chdir(socket_dir)
        assert resolve_unix_socket_path("proxy.sock") == os.path.join(os.getcwd(), "proxy.sock")

    def test_tilde_is_expanded_before_measuring(self, socket_dir: Path, monkeypatch):
        monkeypatch.setenv("HOME", str(socket_dir))
        assert resolve_unix_socket_path("~/proxy.sock") == os.path.join(
            os.path.abspath(socket_dir), "proxy.sock"
        )
        # Short as typed, too long once the home directory is substituted.
        monkeypatch.setenv("HOME", str(socket_dir / ("h" * max_socket_path_bytes())))
        with pytest.raises(UnixSocketUnusableError, match="bytes, longer than"):
            resolve_unix_socket_path("~/p.sock")

    def test_empty_path_rejected(self):
        with pytest.raises(UnixSocketUnusableError, match="not an empty string"):
            resolve_unix_socket_path("")


class TestProxyConfigUds:
    def test_instance_key_is_port_for_tcp(self):
        assert ProxyConfig(port=9123).instance_key == 9123

    def test_instance_key_distinguishes_socket_paths(self):
        first = ProxyConfig(uds="/run/a/proxy.sock").instance_key
        second = ProxyConfig(uds="/run/b/proxy.sock").instance_key
        assert isinstance(first, str) and first.startswith("uds-")
        assert first != second
        assert first != ProxyConfig().instance_key

    def test_empty_path_rejected(self):
        with pytest.raises(ValueError, match="uds"):
            ProxyConfig(uds="")


class TestCliUdsFlag:
    def test_banner_and_config_use_the_resolved_path(self, socket_dir: Path, monkeypatch):
        monkeypatch.chdir(socket_dir)
        result, config = self._invoke(
            ["--uds", "proxy.sock"], env={"HEADROOM_HOST": None, "HEADROOM_PORT": None}
        )
        resolved = os.path.join(os.getcwd(), "proxy.sock")
        assert result.exit_code == 0, result.output
        assert config is not None
        assert config.uds == resolved
        assert f"unix:{resolved}" in result.output
        assert f"curl --unix-socket {resolved} http://localhost/health" in result.output

    def test_too_long_path_is_a_one_line_error_before_the_banner(self, socket_dir: Path):
        path = str(socket_dir / ("a" * 300))
        result, config = self._invoke(
            ["--uds", path], env={"HEADROOM_HOST": None, "HEADROOM_PORT": None}
        )
        assert result.exit_code == 1
        assert config is None
        assert result.output.count("\n") == 1
        assert result.output.startswith(f"Error: socket path {path} is {len(path)} bytes")
        assert "unix:" not in result.output

    def _invoke(self, args: list[str], env: dict[str, str | None] | None = None):
        captured: dict[str, ProxyConfig] = {}

        def fake_run_server(config, **kwargs):
            captured["config"] = config

        with patch("headroom.proxy.server.run_server", fake_run_server):
            result = CliRunner().invoke(main, ["proxy", *args], env=env or {})
        return result, captured.get("config")

    def test_uds_reaches_proxy_config(self):
        result, config = self._invoke(
            ["--uds", "/run/x/proxy.sock"], env={"HEADROOM_HOST": None, "HEADROOM_PORT": None}
        )
        assert result.exit_code == 0, result.output
        assert config is not None
        assert config.uds == "/run/x/proxy.sock"
        assert "unix:/run/x/proxy.sock" in result.output

    @pytest.mark.parametrize(
        ("args", "env"),
        [
            (["--uds", "/run/x/proxy.sock", "--port", "9000"], {"HEADROOM_PORT": None}),
            (["--uds", "/run/x/proxy.sock", "--host", "127.0.0.1"], {"HEADROOM_HOST": None}),
            (["--uds", "/run/x/proxy.sock"], {"HEADROOM_PORT": "9000"}),
        ],
    )
    def test_uds_with_tcp_address_refused(self, args, env):
        result, config = self._invoke(args, env=env)
        assert result.exit_code == 2
        assert "--uds cannot be combined with" in result.output
        assert config is None


def _env_without_tcp_listen_vars() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if k not in ("HEADROOM_HOST", "HEADROOM_PORT")}


class TestCliSocketInUse:
    def test_live_listener_reported_as_usage_error(self, socket_dir: Path):
        path = socket_dir / "proxy.sock"
        live = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        live.bind(str(path))
        live.listen()

        def run_server_binding(config, **kwargs):
            bind_unix_listener(config.uds)

        try:
            with patch("headroom.proxy.server.run_server", run_server_binding):
                result = CliRunner().invoke(
                    main,
                    ["proxy", "--uds", str(path)],
                    env={"HEADROOM_HOST": None, "HEADROOM_PORT": None},
                )
        finally:
            live.close()
        assert result.exit_code == 1
        assert f"Error: another process is already listening on {path}" in result.output


class TestModuleEntrypointUdsFlag:
    def test_too_long_path_refused(self, socket_dir: Path):
        path = str(socket_dir / ("a" * 300))
        proc = subprocess.run(
            [sys.executable, "-m", "headroom.proxy.server", "--uds", path],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            env=_env_without_tcp_listen_vars(),
        )
        assert proc.returncode == 1
        assert proc.stderr.strip().startswith(f"error: socket path {path} is {len(path)} bytes")

    def test_uds_with_port_refused(self):
        proc = subprocess.run(
            [sys.executable, "-m", "headroom.proxy.server", "--uds", "/run/x.sock", "--port", "9"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            env=_env_without_tcp_listen_vars(),
        )
        assert proc.returncode == 2
        assert "not allowed with argument" in proc.stderr

    def test_uds_with_port_env_refused(self):
        proc = subprocess.run(
            [sys.executable, "-m", "headroom.proxy.server", "--uds", "/run/x.sock"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            env={**_env_without_tcp_listen_vars(), "HEADROOM_PORT": "9000"},
        )
        assert proc.returncode == 1
        assert "--uds cannot be combined with HEADROOM_PORT" in proc.stderr


# Python on Windows defines no socket.AF_UNIX. The tests remove the attribute from the socket module rather than run on Windows; everything that uses it is imported first, so only the guard sees it missing.
_NO_AF_UNIX_SCRIPT = textwrap.dedent(
    """
    import runpy, socket, sys
    import headroom.proxy.server
    del socket.AF_UNIX
    sys.argv = ["headroom.proxy.server", "--uds", "proxy.sock"]
    runpy.run_module("headroom.proxy.server", run_name="__main__")
    """
)


class TestPlatformWithoutUnixSockets:
    def test_require_unix_sockets_names_the_platform(self, monkeypatch):
        monkeypatch.delattr(socket, "AF_UNIX")
        monkeypatch.setattr(sys, "platform", "win32")
        with pytest.raises(UnixSocketUnusableError, match="win32 does not provide"):
            require_unix_sockets()

    def test_cli_refuses_before_any_other_work(self, monkeypatch):
        monkeypatch.delattr(socket, "AF_UNIX")

        def fail(*args, **kwargs):
            raise AssertionError("work ran before the AF_UNIX guard")

        monkeypatch.setattr("headroom.cli.proxy._reexec_with_malloc_tuning", fail)
        monkeypatch.setattr("headroom.cli.proxy.ensure_proxy_dependencies", fail)
        result = CliRunner().invoke(
            main,
            ["proxy", "--uds", "proxy.sock"],
            env={"HEADROOM_HOST": None, "HEADROOM_PORT": None},
        )
        assert result.exit_code == 1
        assert "Error: --uds needs unix domain sockets" in result.output

    def test_module_entrypoint_refuses(self):
        proc = subprocess.run(
            [sys.executable, "-W", "ignore::RuntimeWarning", "-c", _NO_AF_UNIX_SCRIPT],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            env=_env_without_tcp_listen_vars(),
        )
        assert proc.returncode == 1
        assert proc.stderr.strip().startswith("error: --uds needs unix domain sockets")
