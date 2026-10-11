"""Whether another user could change a file or directory the proxy trusts.

Shared by the ``--uds`` socket's parent-directory check and the upstream allowlist file, which both trust a path only if nobody but the user running the proxy can replace what it names. Standard library only, so the SSRF guard can import it without widening its own dependencies. POSIX only: ownership and mode bits are what is judged, so callers on a platform without ``os.geteuid`` do not call it.
"""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass


@dataclass(frozen=True)
class ForeignWriteAccess:
    """Why someone other than the current effective user could change what a ``stat`` describes.

    ``description`` completes "<subject> is ...", for example ``owned by uid 0, not by the user running the proxy (uid 501)``. ``other_owner`` tells the two causes apart for callers that advise differently on each.
    """

    description: str
    other_owner: bool


def foreign_write_access(info: os.stat_result) -> ForeignWriteAccess | None:
    """Return why another user could change the file or directory ``info`` describes, else ``None``.

    Its owner can always change it, so one that is not the current effective user is refused first; otherwise group or other write permission is. For a directory, write permission is what allows unlinking or renaming its entries, whoever owns them, so either lets someone else replace what the directory holds.
    """
    if info.st_uid != os.geteuid():
        return ForeignWriteAccess(
            f"owned by uid {info.st_uid}, not by the user running the proxy (uid {os.geteuid()})",
            other_owner=True,
        )
    if info.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        return ForeignWriteAccess(
            f"writable by group or others (mode {stat.S_IMODE(info.st_mode):04o})",
            other_owner=False,
        )
    return None
