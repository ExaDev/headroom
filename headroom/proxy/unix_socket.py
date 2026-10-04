"""Unix domain socket listener for the proxy.

``headroom proxy --uds PATH`` serves HTTP on a unix socket instead of a TCP host and port. A socket inside a directory only its owner can enter authenticates every peer by filesystem ownership, so a forwarder never sends credentials to whichever local process happens to hold a free loopback port.

The socket is bound here and handed to uvicorn as a file descriptor rather than passed as uvicorn's ``uds`` argument, because uvicorn's own path:

* chmods the socket to ``0o666`` (or copies the mode of whatever file was at the path), so any local user who can reach the directory can connect;
* lets asyncio silently unlink any socket already at the path, including one a live server is still listening on;
* fails outright in multi-worker mode when the path exists, even when the socket there is stale.

Contract with the caller:

* The parent directory must already exist and must be private (mode ``0o700``, owned by the user running the proxy). This module never creates or chmods it; the socket's own ``0o600`` mode is the second layer, not the only one.
* A socket file left at the path by a process that has exited (nothing accepts connections on it) is removed and replaced. A socket that still accepts connections, or any path that is not a socket, is refused with :class:`UnixSocketInUseError`.
* Checking for a stale socket and binding are two steps, so two proxies started on the same path at the same instant can race. Supervising a single proxy per path is the caller's job.
* On clean shutdown the socket file is removed, but only if the path still names the socket this process bound. That includes SIGTERM: uvicorn finishes its graceful shutdown and then re-raises the signal with the previous handler restored, which under the default handler would kill the process before any cleanup ran, so :func:`serving_unix_socket` installs a handler that unwinds the stack instead and re-raises SIGTERM itself once the file is gone.
"""

from __future__ import annotations

import errno
import os
import signal
import socket
import stat
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from types import FrameType

#: Owner read and write. Connecting to a unix socket needs write permission on it, so this admits only the proxy's own user.
SOCKET_MODE = stat.S_IRUSR | stat.S_IWUSR


class UnixSocketInUseError(RuntimeError):
    """The socket path is held by a live listener or by something that is not a socket."""


class UnixSocketUnusableError(RuntimeError):
    """A unix socket cannot be served here: the platform has none, or the requested path cannot hold one."""


def require_unix_sockets() -> None:
    """Raise :class:`UnixSocketUnusableError` on a platform without ``socket.AF_UNIX``.

    Windows builds of Python define no ``AF_UNIX``, so ``--uds`` would otherwise die with an ``AttributeError`` the first time the socket module is asked for it. Entry points call this before doing any other work.
    """
    if not hasattr(socket, "AF_UNIX"):
        raise UnixSocketUnusableError(
            f"--uds needs unix domain sockets, which {sys.platform} does not provide"
        )


@dataclass
class UnixSocketListener:
    """A bound, not yet listening, unix socket and the identity of the file it created."""

    path: str
    sock: socket.socket
    device: int
    inode: int

    def close(self) -> None:
        """Close the socket and remove its file if the path still names this socket.

        A successor that replaced the stale-looking file after this process stopped accepting is left alone.
        """
        self.sock.close()
        try:
            current = os.lstat(self.path)
        except FileNotFoundError:
            return
        if (current.st_dev, current.st_ino) == (self.device, self.inode):
            os.unlink(self.path)


def _remove_stale_socket(path: str) -> None:
    """Remove a socket file at *path* that nothing is listening on.

    Raises :class:`UnixSocketInUseError` when *path* is not a socket or a listener still accepts connections on it.
    """
    try:
        existing = os.lstat(path)
    except FileNotFoundError:
        return
    if not stat.S_ISSOCK(existing.st_mode):
        raise UnixSocketInUseError(f"{path} exists and is not a socket; refusing to replace it")
    probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    # Non-blocking so a live listener with a full backlog reports EAGAIN instead of hanging startup.
    probe.setblocking(False)
    try:
        result = probe.connect_ex(path)
    finally:
        probe.close()
    if result == errno.ECONNREFUSED:
        os.unlink(path)
        return
    if result in (0, errno.EAGAIN, errno.EINPROGRESS):
        raise UnixSocketInUseError(f"another process is already listening on {path}")
    raise OSError(result, os.strerror(result), path)


def bind_unix_listener(path: str) -> UnixSocketListener:
    """Bind a stream socket at *path* with mode :data:`SOCKET_MODE`.

    The socket is chmodded before uvicorn calls ``listen()`` on it, so no peer can connect while the file still carries the umask-derived mode.
    """
    _remove_stale_socket(path)
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.bind(path)
        os.chmod(path, SOCKET_MODE)
        bound = os.lstat(path)
    except BaseException:
        sock.close()
        raise
    return UnixSocketListener(path=path, sock=sock, device=bound.st_dev, inode=bound.st_ino)


class _Terminated(BaseException):
    """Raised from the SIGTERM handler; a BaseException so ``except Exception`` blocks cannot swallow it."""


def _raise_terminated(signum: int, frame: FrameType | None) -> None:
    raise _Terminated


@contextmanager
def serving_unix_socket(path: str) -> Iterator[UnixSocketListener]:
    """Bind *path* for the duration of the block, removing the socket file when it exits.

    On the main thread SIGTERM unwinds the block, the file is removed, and SIGTERM is then re-raised under the handler that was in place before, so the process still ends the way it would have without this listener. ``signal.signal`` only works on the main thread, and off it uvicorn neither captures nor re-raises signals, so no handler is installed there.
    """
    listener = bind_unix_listener(path)
    on_main_thread = threading.current_thread() is threading.main_thread()
    previous = signal.signal(signal.SIGTERM, _raise_terminated) if on_main_thread else None
    terminated = False
    try:
        yield listener
    except _Terminated:
        terminated = True
    finally:
        if on_main_thread:
            signal.signal(signal.SIGTERM, previous)
        listener.close()
    if terminated:
        signal.raise_signal(signal.SIGTERM)
