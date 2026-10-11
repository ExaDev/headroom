"""SSRF guard for client-supplied upstream base URLs (WEB-01).

Clients may redirect the proxy's upstream via the ``x-headroom-base-url`` header
(BYOK / custom OpenAI-compatible endpoints). Without validation this lets a
caller turn the proxy into a confused deputy — reaching cloud-metadata
(``169.254.169.254``) or internal RFC1918 hosts the caller cannot reach directly.

Policy:
  * Default: reject destinations that resolve to private, loopback, link-local,
    or otherwise non-public addresses. Public hosts (api.openai.com, api.x.ai,
    Azure, ...) are allowed so ordinary BYOK keeps working.
  * When ``HEADROOM_ALLOWED_BASE_URLS`` is set (hosts or URLs separated by
    commas, newlines, carriage returns or tabs), bare hosts permit every safe scheme/port for that
    host, while URLs permit only their exact normalized origin. A bare host
    with a port (``127.0.0.1:4100``) is neither and is skipped as invalid.
    Because that is an explicit operator choice, allowlisted destinations may
    point at internal/on-prem endpoints.
  * ``HEADROOM_ALLOWED_BASE_URLS_FILE`` (or ``headroom proxy
    --allowed-base-urls-file``) names a file holding more entries in the same
    format, plus ``#`` comment lines and an optional byte-order mark, merged with the variable's. The variable is fixed for the
    life of the process; the file is re-read whenever it changes, so a
    supervisor whose own listed endpoint moves (a loopback port chosen at
    start-up, say) can rewrite the file instead of restarting a proxy that
    sessions are still using. Replace it atomically (write a sibling, then
    rename over it): a change is detected by the file's identity (device,
    inode, size, and modification and status-change times), and an unchanged
    file costs one ``stat`` per check, never a read.

Naming a file is itself the operator's choice of allowlist mode, so it never
falls back to the public-host default, whatever state the file is in:

  * readable, with entries: allowlist mode, admitting the file's entries plus
    the variable's.
  * readable but empty (or holding no valid entry): allowlist mode admitting
    only the variable's entries, and nothing at all when the variable is unset.
    This follows the existing meaning of the two shapes ``allowlisted_destinations``
    returns: ``None`` is "no allowlist, apply the default policy", while an
    empty list is "allowlist mode, nothing listed", which a variable holding
    only separators already produces. A supervisor that truncates the file
    before it knows its endpoints therefore gets a proxy that refuses every
    caller-supplied upstream, not one that briefly admits every public host.
  * missing or unusable (permissions, not UTF-8, not a regular file, larger
    than ``_MAX_ALLOWLIST_FILE_BYTES``): the same as empty, logged at warning level once per distinct failure rather than on
    every request, and logged again if it recurs after the file was read.
  * the file variable set to blank: as if unset, mirroring how a blank
    ``HEADROOM_ALLOWED_BASE_URLS`` is treated.

Answering "is this safe?" is only half the job: the answer has to survive to the
socket. Every accepted destination therefore has the addresses it was judged on
recorded against *the request that asked* (see :func:`guarded_pin`), and the
proxy's HTTP client dials one of *those* instead of resolving the name a second
time — ``headroom/proxy/upstream_pinning.py`` is the connect-time half.

The record is a :mod:`contextvars` scope, not a process-global cache, and that
distinction is the safety property rather than a tidiness one. A hostname-keyed
cache can only answer "is there a pin for this name right now?", so an absent
one is indistinguishable from a name that was never guarded — which means a
connection for a guarded destination whose pin is missing falls back to ordinary
DNS, the exact re-resolution this module exists to prevent. A scope answers the
question that actually matters, "was *this* request's destination guarded?", so
a guarded destination with no usable pin is a denial. It also stops one caller's
pin from being visible to another caller's connection.

This module intentionally depends only on the standard library so it is safe to
import from any handler without risking an import cycle.
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import os
import re
import socket
import stat
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as _FutureTimeout
from contextvars import ContextVar
from dataclasses import dataclass
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

ALLOWED_BASE_URLS_ENV = "HEADROOM_ALLOWED_BASE_URLS"
ALLOWED_BASE_URLS_FILE_ENV = "HEADROOM_ALLOWED_BASE_URLS_FILE"

# `socket.getaddrinfo` has no timeout parameter and runs on whatever thread
# calls it -- which, for the proxy, is the event loop. A caller-supplied host
# that resolves slowly therefore stalls every other in-flight request, so the
# lookup is bounded here and fails closed when it overruns. Callers already in
# async context should prefer `is_safe_upstream_url_async`, which keeps the
# wait off the loop entirely.
RESOLVE_TIMEOUT_ENV = "HEADROOM_UPSTREAM_RESOLVE_TIMEOUT_S"
_DEFAULT_RESOLVE_TIMEOUT_S = 3.0
_RESOLVER_POOL = ThreadPoolExecutor(max_workers=8, thread_name_prefix="hr-upstream-dns")


def _resolve_timeout_seconds() -> float:
    raw = (os.environ.get(RESOLVE_TIMEOUT_ENV) or "").strip()
    if not raw:
        return _DEFAULT_RESOLVE_TIMEOUT_S
    try:
        value = float(raw)
    except ValueError:
        return _DEFAULT_RESOLVE_TIMEOUT_S
    return value if value > 0 else _DEFAULT_RESOLVE_TIMEOUT_S


_SAFE_SCHEMES = {"http", "https", "ws", "wss"}

# Pinning the addresses a check was made against.
#
# `socket.getaddrinfo` below and the HTTP client's own lookup at connect time
# are two separate resolutions of the same name, and an attacker who controls
# its authoritative DNS answers them differently: a public address for the
# check, 169.254.169.254 (or an RFC1918 host) for the connection. That is DNS
# rebinding, and it makes a verdict based only on the first answer advisory --
# the socket never sees it. So the addresses that were actually judged are
# recorded for the connection to dial.
#
# The record lives in a contextvar, which scopes it to the request that made the
# check: a task inherits its creating context, so the pin an `async def` route
# writes is the pin its own upstream call reads, and no other caller's. The
# entry is also the *marker* that this destination went through the guard --
# see `guarded_pin`.
#
# The TTL bounds how stale a judged address may be by the time a socket opens.
# It is short on purpose and deliberately not configurable: a pin only has to
# bridge a check and the connection it authorises, while an operator who
# stretched it to hours would be authorising a connection on a DNS answer from
# hours ago. Running out is NOT a fallback to ordinary resolution -- the whole
# point is that the destination is known to be caller-supplied, so the
# connect-time layer denies rather than resolving the attacker's name again.
_PIN_TTL_SECONDS = 60.0
# Hostnames are caller-supplied, so a scope is bounded even though it dies with
# the request that owns it. Overflow rejects the destination rather than leaving
# it unpinned, since an unpinned entry is one the connect layer would have to
# resolve itself. A request guards one or two upstreams; this is not reachable
# by any legitimate route.
_PIN_MAX_ENTRIES = 64


@dataclass(frozen=True)
class UpstreamPin:
    """The addresses one guarded destination was judged on, and when they lapse.

    An instance existing at all is the statement "this request had this host
    validated by the guard". ``addresses`` is what the check accepted;
    ``expires_at`` is a :func:`time.monotonic` deadline past which the answer is
    too old to authorise a socket.
    """

    addresses: tuple[str, ...]
    expires_at: float

    def is_expired(self, now: float | None = None) -> bool:
        return (time.monotonic() if now is None else now) >= self.expires_at


_PIN_SCOPE: ContextVar[dict[str, UpstreamPin] | None] = ContextVar(
    "headroom_upstream_pin_scope", default=None
)


def begin_pin_scope() -> dict[str, UpstreamPin]:
    """Create (or return) this context's pin scope. Returns the live mapping.

    Call this on the thread/task that will *make* the request, before handing
    the check off anywhere else. A contextvar ``set`` inside a worker thread
    lands in that thread's copy of the context and is lost when it returns,
    so :func:`is_safe_upstream_url_async` establishes the mapping here and then
    mutates it from the worker -- the mapping object is shared, the binding is
    not.
    """
    scope = _PIN_SCOPE.get()
    if scope is None:
        scope = {}
        _PIN_SCOPE.set(scope)
    return scope


def _record_validated_addresses(
    host: str,
    addresses: tuple[str, ...],
    scope: dict[str, UpstreamPin] | None = None,
) -> bool:
    """Remember what ``host`` was judged on, for the connection that follows.

    Returns False when the scope is full, which the caller turns into a
    rejection: a destination we cannot pin is one the connection would resolve
    for itself.
    """
    if not host or not addresses:
        return False
    if scope is None:
        scope = begin_pin_scope()
    if host not in scope and len(scope) >= _PIN_MAX_ENTRIES:
        return False
    scope[host] = UpstreamPin(addresses, time.monotonic() + _PIN_TTL_SECONDS)
    return True


def guarded_pin(host: str) -> UpstreamPin | None:
    """Return this request's pin for ``host``, expired or not, else ``None``.

    ``None`` means "the current request never had this destination checked" --
    a configured provider upstream, a proxy endpoint, an allowlisted host --
    and such a destination resolves normally, exactly as it always has.

    A returned pin means the opposite, and it says so even when it has expired.
    That is the whole reason expiry does not delete the entry: the connect-time
    layer has to be able to tell "never guarded" from "guarded, but the judged
    addresses are too old to use", because only the first of those is safe to
    resolve by name.
    """
    scope = _PIN_SCOPE.get()
    if not scope:
        return None
    key = (host or "").strip().lower()
    if not key:
        return None
    return scope.get(key)


def validated_addresses(host: str) -> tuple[str, ...] | None:
    """Return the still-valid addresses this request's check accepted for ``host``.

    ``None`` covers both "never guarded here" and "guarded but lapsed", so it is
    not a safe basis for a connect decision -- use :func:`guarded_pin`, which
    keeps those apart. Retained for callers that only want the addresses.
    """
    pin = guarded_pin(host)
    if pin is None or pin.is_expired():
        return None
    return pin.addresses


def clear_validated_addresses() -> None:
    """Drop this context's pins. For tests, and between reused contexts."""
    _PIN_SCOPE.set(None)


Destinations = tuple[frozenset[str], frozenset[tuple[str, str, int]]]
"""Bare hosts (any safe scheme and port) and exact ``(scheme, host, port)`` origins."""

_NO_DESTINATIONS: Destinations = (frozenset(), frozenset())


@dataclass(frozen=True)
class _FileIdentity:
    """What distinguishes one version of a file from another without reading it.

    A rename-replaced file has a new inode (and, across a mount, a new device). An in-place rewrite changes the modification time, but that time is settable (``cp -p``, ``touch -r``, ``os.utime``) and coarse on some filesystems, so a same-size rewrite can leave it as it was. The status-change time cannot be set by a writer and moves on every write, ``chmod`` and ``utime``, so it catches both that rewrite and a permission change that should stop the file being read. Nanosecond fields, so two changes within one coarse tick still differ where the filesystem records nanoseconds.
    """

    device: int
    inode: int
    size: int
    mtime_ns: int
    ctime_ns: int

    @classmethod
    def of(cls, st: os.stat_result) -> _FileIdentity:
        return cls(st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)


# The longest entry worth listing is a URL origin: "https://", a 253-character DNS name (or a bracketed IPv6 literal, shorter), ":65535" and a separator, about 270 bytes. One MiB holds several thousand of those, orders of magnitude beyond any operator's allowlist, while bounding the memory and parse work one reload can cost. A file past it is a wrong path (a log, a binary) rather than a list, and it is refused whole: reading only a prefix could turn it into a different, valid-looking list.
_MAX_ALLOWLIST_FILE_BYTES = 1 << 20

# `os.open` flags: O_NONBLOCK keeps a FIFO that appears at the path between the `stat` and the open from blocking the open until a writer arrives, and O_NOCTTY keeps a terminal that appears there from becoming the process's controlling terminal. Windows has neither flag, nor FIFOs or controlling terminals.
_O_NONBLOCK = getattr(os, "O_NONBLOCK", 0)
_O_NOCTTY = getattr(os, "O_NOCTTY", 0)


class _UnusableAllowlistFile(Exception):
    """The file at the path cannot be used as an allowlist, and which version of it that was."""

    def __init__(self, reason: str, identity: _FileIdentity) -> None:
        super().__init__(reason)
        self.identity = identity


def _read_allowlist_file(path: str, st: os.stat_result) -> tuple[_FileIdentity, str]:
    """Return the identity of the file read and its text; ``st`` is the caller's ``stat`` of it.

    Raises :class:`_UnusableAllowlistFile` when it is not a regular file, cannot be opened or read, is larger than ``_MAX_ALLOWLIST_FILE_BYTES`` or is not UTF-8. The type is checked on ``st`` before anything is opened, so a device or terminal named by mistake is never opened, and again on the descriptor, for one swapped in after the ``stat``. The size is checked on the descriptor before anything is read.
    """
    identity = _FileIdentity.of(st)
    if not stat.S_ISREG(st.st_mode):
        raise _UnusableAllowlistFile("not a regular file", identity)
    try:
        fd = os.open(path, os.O_RDONLY | _O_NONBLOCK | _O_NOCTTY)
    except OSError as exc:
        raise _UnusableAllowlistFile(str(exc), identity) from exc
    with os.fdopen(fd, "rb") as handle:
        st = os.fstat(handle.fileno())
        # The identity recorded is the descriptor's, not the caller's `stat`'s. A rename between the two means the bytes read are the newer file's; filed under the older identity, the next check would see a mismatch and read the file again for nothing.
        identity = _FileIdentity.of(st)
        if not stat.S_ISREG(st.st_mode):
            raise _UnusableAllowlistFile("not a regular file", identity)
        too_large = _UnusableAllowlistFile(
            f"larger than {_MAX_ALLOWLIST_FILE_BYTES} bytes", identity
        )
        if st.st_size > _MAX_ALLOWLIST_FILE_BYTES:
            raise too_large
        try:
            # One byte past the cap, so a file that grew after the `fstat` is still caught.
            data = handle.read(_MAX_ALLOWLIST_FILE_BYTES + 1)
        except OSError as exc:
            raise _UnusableAllowlistFile(str(exc), identity) from exc
    if len(data) > _MAX_ALLOWLIST_FILE_BYTES:
        raise too_large
    try:
        # utf-8-sig drops the byte-order mark some editors write; without it, the mark would join the first entry and make it invalid.
        return identity, data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise _UnusableAllowlistFile(f"not UTF-8 ({exc})", identity) from exc


class _AllowlistFiles:
    """What each allowlist file held, re-read only when the file's identity changes.

    A version that could not be used is cached too, as ``None``, so a file that is oversized, not UTF-8 or unreadable costs one ``stat`` per check like a good one, not an open and a read on every request.

    Thread-safe: the event loop and the resolver pool both reach the guard. Checks against an unchanged file take only the bookkeeping lock. Checks that see a change queue on one reload lock and re-check the cache once they hold it, so a change is read and logged once however many requests arrive together; the wait is bounded by one read of at most ``_MAX_ALLOWLIST_FILE_BYTES``.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._reload_lock = threading.Lock()
        self._loaded: dict[str, tuple[_FileIdentity, Destinations | None]] = {}
        self._reported: dict[str, str] = {}

    def _cached(self, path: str, identity: _FileIdentity) -> tuple[bool, Destinations | None]:
        """``(True, what that version held)`` if ``identity`` is the version cached, else ``(False, None)``."""
        with self._lock:
            cached = self._loaded.get(path)
        if cached is not None and cached[0] == identity:
            return True, cached[1]
        return False, None

    def load(self, path: str, file_env: str, env: str) -> Destinations | None:
        """Return ``path``'s entries, or ``None`` when it cannot be used.

        ``file_env`` names the variable that pointed here and ``env`` the list
        its entries are merged with; both are for the log lines only.
        """
        try:
            st = os.stat(path)
        except OSError as exc:
            self._report(path, str(exc), file_env, env)
            return None
        identity = _FileIdentity.of(st)
        hit, entries = self._cached(path, identity)
        if hit:
            return entries
        with self._reload_lock:
            # Another check may have read this very version while this one queued.
            hit, entries = self._cached(path, identity)
            if hit:
                return entries
            try:
                identity, raw = _read_allowlist_file(path, st)
            except _UnusableAllowlistFile as exc:
                with self._lock:
                    self._loaded[path] = (exc.identity, None)
                self._report(path, str(exc), file_env, env)
                return None
            entries, skipped = _parse_destinations(raw, comments=True)
            with self._lock:
                self._loaded[path] = (identity, entries)
                recovered = self._reported.pop(path, None) is not None
        logger.log(
            logging.WARNING if skipped else logging.INFO,
            "%s: %s %s, %d host(s) and %d origin(s), %d invalid entr%s skipped",
            file_env,
            "read again after failing:" if recovered else "loaded",
            path,
            len(entries[0]),
            len(entries[1]),
            skipped,
            "y" if skipped == 1 else "ies",
        )
        return entries

    def _report(self, path: str, reason: str, file_env: str, env: str) -> None:
        """Log a failure once, until it changes or the file is read again."""
        with self._lock:
            if self._reported.get(path) == reason:
                return
            self._reported[path] = reason
        logger.warning(
            "%s: cannot read %s (%s). Until it can be read, only the entries in "
            "%s are allowlisted (none when it is unset) and every other "
            "x-headroom-base-url is rejected with 400.",
            file_env,
            path,
            reason,
            env,
        )


_ALLOWLIST_FILES = _AllowlistFiles()

_SEPARATORS = re.compile(r"[,\n\r\t]")

# A comment in the file: a "#" at the very start, or after a space, tab, newline or carriage return, through to the end of its line (the next newline or carriage return, the line-ending members of `_SEPARATORS`). A "#" anywhere else is part of an entry, which then fails to parse as one.
_COMMENT = re.compile(r"(?:^|(?<=[ \t\r\n]))#[^\r\n]*")


def _parse_entry(item: str) -> str | tuple[str, str, int] | None:
    """Return a bare host, an exact ``(scheme, host, port)`` origin, or ``None`` for neither."""
    if any(character.isspace() for character in item):
        # A host or URL never contains whitespace, so "a.example b.example" or "gw.example # prod" in a variable is one entry that could never match anything. Counted as invalid, it is at least reported.
        return None
    try:
        if "://" not in item:
            bare = urlparse(f"//{item}")
            if bare.port is not None:
                # A bare host admits every port, so "127.0.0.1:4100" read that way would admit far more than it names, and read as an origin it would have to guess the scheme. It is neither: the entry is invalid, and an exact origin is written as a URL.
                return None
            return bare.hostname.lower() if bare.hostname else None
        parsed = urlparse(item)
        scheme = parsed.scheme.lower()
        if scheme not in _SAFE_SCHEMES or not parsed.hostname:
            return None
        port = parsed.port
    except ValueError:
        # `urlparse` raises for an unbalanced IPv6 bracket ("[oops", "http://[::1") and for a netloc that NFKC normalisation would turn into URL syntax; `.port` raises for a non-numeric or out-of-range port. Each is one bad entry, skipped like any other, rather than an exception that takes every other entry, and every request, down with it.
        return None
    if port is None:
        port = 443 if scheme in {"https", "wss"} else 80
    return scheme, parsed.hostname.lower(), port


def _parse_destinations(raw: str, *, comments: bool = False) -> tuple[Destinations, int]:
    """Parse hosts and URLs separated by commas, newlines, carriage returns or tabs.

    With ``comments``, a ``#`` at the start of the text or after a space, tab, newline or carriage return begins a comment that runs to the end of its line. Only the file takes comments: a variable has no lines to comment out, and there a ``#`` stays part of an entry, which is then invalid.

    Returns the destinations and how many non-blank entries were skipped as invalid: neither a host nor a URL with a safe scheme, or not parseable at all. A list of nothing valid is an empty allowlist, not an absent one.
    """
    hosts: set[str] = set()
    origins: set[tuple[str, str, int]] = set()
    skipped = 0
    if comments:
        raw = _COMMENT.sub("", raw)
    for item in _SEPARATORS.split(raw):
        item = item.strip()
        if not item:
            continue
        entry = _parse_entry(item)
        if entry is None:
            skipped += 1
        elif isinstance(entry, str):
            hosts.add(entry)
        else:
            origins.add(entry)
    return (frozenset(hosts), frozenset(origins)), skipped


_SKIPPED_IN_VARIABLE: dict[str, str] = {}
_SKIPPED_IN_VARIABLE_LOCK = threading.Lock()


def _parse_variable(env: str, raw: str) -> Destinations:
    """Parse ``env``'s value, warning once per distinct value that holds invalid entries.

    The variable is parsed on every check, so the warning is keyed on the value rather than logged per request.
    """
    destinations, skipped = _parse_destinations(raw)
    if skipped:
        with _SKIPPED_IN_VARIABLE_LOCK:
            already = _SKIPPED_IN_VARIABLE.get(env) == raw
            _SKIPPED_IN_VARIABLE[env] = raw
        if not already:
            logger.warning(
                "%s: skipped %d invalid entr%s", env, skipped, "y" if skipped == 1 else "ies"
            )
    return destinations


def allowlisted_destinations(
    env: str = ALLOWED_BASE_URLS_ENV,
    file_env: str | None = ALLOWED_BASE_URLS_FILE_ENV,
) -> Destinations | None:
    """Return the operator's allowlist from ``env`` and the file ``file_env`` names.

    ``None`` means no allowlist at all: neither variable is set (or each is blank), and the caller applies its default policy. Anything else is allowlist mode, possibly with nothing listed. Once ``file_env`` names a file the result is never ``None``, even when the file is empty, missing or unreadable; see the module docstring for why. Pass ``file_env=None`` for a list that has no file form.
    """
    raw = os.environ.get(env)
    listed = _parse_variable(env, raw) if raw and raw.strip() else None
    if file_env is None:
        return listed
    path = (os.environ.get(file_env) or "").strip()
    if not path:
        return listed
    from_file = _ALLOWLIST_FILES.load(path, file_env, env) or _NO_DESTINATIONS
    if listed is None:
        return from_file
    return listed[0] | from_file[0], listed[1] | from_file[1]


def destination_listed(
    allow: Destinations,
    scheme: str,
    host: str,
    port: int | None,
) -> bool:
    """Whether a destination matches a parsed host/origin list.

    A bare host admits every port for that host; a URL admits only its exact
    origin, with the scheme's default port filled in when ``port`` is ``None``.
    """
    hosts, origins = allow
    if host.lower() in hosts:
        return True
    scheme = scheme.lower()
    if port is None:
        port = 443 if scheme in {"https", "wss"} else 80
    return (scheme, host.lower(), port) in origins


# RFC 6052 / RFC 8215: these IPv6 prefixes embed an IPv4 address in their low
# 32 bits, and `ipaddress` reports the well-known one as globally routable. On a
# NAT64 network `64:ff9b::7f00:1` reaches 127.0.0.1, so the embedded address is
# what has to be judged. 6to4, Teredo and IPv4-mapped forms are already caught
# by the `is_global` test below.
_NAT64_PREFIXES = (
    ipaddress.IPv6Network("64:ff9b::/96"),
    ipaddress.IPv6Network("64:ff9b:1::/48"),
)


def _nat64_embedded_ipv4(addr: ipaddress.IPv6Address) -> ipaddress.IPv4Address | None:
    if not any(addr in prefix for prefix in _NAT64_PREFIXES):
        return None
    try:
        return ipaddress.IPv4Address(int(addr) & 0xFFFFFFFF)
    except (ipaddress.AddressValueError, ValueError):  # pragma: no cover - defensive
        return None


def _is_internal_address(ip: str) -> bool:
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return True  # unparseable (e.g. scoped link-local) -> treat as unsafe
    if (
        addr.is_private
        or addr.is_loopback
        or addr.is_link_local
        or addr.is_reserved
        or addr.is_multicast
        or addr.is_unspecified
    ):
        return True
    # Anything not globally routable. This is what catches RFC 6598 shared
    # address space (100.64.0.0/10) -- which `is_private` does not flag, and
    # which reaches ISP and cloud-internal infrastructure -- along with
    # benchmarking (198.18/15), TEST-NET, 240/4, 6to4 and Teredo tunnels that
    # embed an internal IPv4, and any future special-use range the stdlib
    # learns about.
    if not addr.is_global:
        return True
    if isinstance(addr, ipaddress.IPv6Address):
        embedded = _nat64_embedded_ipv4(addr)
        if embedded is not None and _is_internal_address(str(embedded)):
            return True
    return False


def is_safe_upstream_url(url: str, *, scope: dict[str, UpstreamPin] | None = None) -> bool:
    """Return True if ``url`` is a safe client-chosen upstream destination.

    In allowlist mode only allowlisted hosts pass. Otherwise the host is
    resolved and rejected if any resolved address is internal/metadata, which
    also catches DNS names that point at private space.

    A destination accepted on its addresses also has them pinned against the
    calling request, so the connection that follows cannot be re-pointed by a
    second DNS answer. Allowlist mode pins nothing: it admits a host by name
    without resolving it at all, precisely so split-horizon and on-prem
    endpoints -- whose addresses are the operator's business, and may
    legitimately move or round-robin -- keep working.

    ``scope`` is the pin mapping to write into; it defaults to the calling
    context's. :func:`is_safe_upstream_url_async` passes its caller's mapping
    explicitly because the check itself runs on a worker thread.
    """
    parsed = urlparse((url or "").strip())
    if parsed.scheme.lower() not in _SAFE_SCHEMES:
        return False
    host = parsed.hostname
    if not host:
        return False

    allow = allowlisted_destinations()
    if allow is not None:
        try:
            port = parsed.port
        except ValueError:
            port = -1  # matches no origin; a listed bare host still passes, as before
        return destination_listed(allow, parsed.scheme, host, port)

    try:
        infos = _RESOLVER_POOL.submit(
            socket.getaddrinfo, host, None, 0, 0, socket.IPPROTO_TCP
        ).result(timeout=_resolve_timeout_seconds())
    except (OSError, _FutureTimeout):
        # Resolution and connection are separate operations, so allowing a DNS
        # miss here would fail open if the name resolves on the later lookup.
        # A lookup that overruns the budget is treated the same way.
        # Operators can explicitly allowlist split-horizon/internal endpoints.
        return False
    addresses = tuple(dict.fromkeys(str(info[4][0]) for info in infos))
    if any(_is_internal_address(address) for address in addresses):
        return False
    # Every address in this answer passed, so pin them all: the connection may
    # dial any one of them and still be dialling something this check accepted.
    # Keeping the whole set (in resolver order) rather than a single winner
    # leaves the connection somewhere to go when the first address is an AAAA on
    # a host with no IPv6 route -- the fallback the OS resolver would otherwise
    # have done for us.
    #
    # A destination that cannot be pinned is rejected rather than allowed
    # through unpinned: an unpinned client-supplied host is one the connection
    # would have to resolve for itself, which is the hole this closes.
    return _record_validated_addresses(host.lower(), addresses, scope)


async def is_safe_upstream_url_async(url: str) -> bool:
    """Async form of :func:`is_safe_upstream_url` for event-loop callers.

    Same policy; the blocking resolution runs off the loop so a hostile or
    slow-resolving hostname cannot stall unrelated in-flight requests.

    The pin scope is opened *here*, on the caller's task, and handed to the
    worker to fill in. Letting the worker open it instead would bind the
    contextvar in the thread's throwaway copy of the context, so the pin would
    vanish before the request it authorises ever reached a socket -- and a
    guarded destination with no pin is a denied one.
    """
    scope = begin_pin_scope()
    return await asyncio.to_thread(is_safe_upstream_url, url, scope=scope)
