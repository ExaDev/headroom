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


def foreign_write_access(
    info: os.stat_result, *, root_may_own: bool = False
) -> ForeignWriteAccess | None:
    """Return why another user could change the file or directory ``info`` describes, else ``None``.

    Its owner can always change it, so an owner other than the current effective user is refused first; otherwise group or other write permission is. For a directory, write permission is what allows unlinking or renaming its entries, whoever owns them, so either lets someone else replace what the directory holds.

    The sticky bit is never a substitute for write permission: in a sticky shared directory anyone can still create a new entry, such as a symlink to a file of their choosing or (where hard links to others' files are allowed) a hard link to one, at a path nobody has claimed yet, and checking the file that entry leads to cannot tell who made it.

    ``root_may_own`` relaxes the owner rule for paths the proxy only reads, never binds; it is off by default, which is the ``--uds`` rule. It also accepts uid 0, which can change anything anyway, so trusting it adds no one, and it lets a proxy running as another user read a root-owned file, such as one in a root-owned bind-mounted directory.
    """
    euid = os.geteuid()
    if info.st_uid != euid and not (root_may_own and info.st_uid == 0):
        owners = f"the user running the proxy (uid {euid})" + (" or root" if root_may_own else "")
        return ForeignWriteAccess(f"owned by uid {info.st_uid}, not by {owners}", other_owner=True)
    if info.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        return ForeignWriteAccess(
            f"writable by group or others (mode {stat.S_IMODE(info.st_mode):04o})",
            other_owner=False,
        )
    return None
