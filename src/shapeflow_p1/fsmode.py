"""File modes for artifacts that cross a uid boundary.

The study runs as five service uids and shares data between them with POSIX **default ACLs**
rather than a common group: ``scripts/install_host.sh`` puts ``d:u:sfrunner:r-x`` on the
steward's frozen corpus, ``d:u:sfevaluator:r-x`` on the runner's object store, runs and
checkpoints, and so on.  A named-user ACL entry is only ever as strong as the file's ACL
*mask*, and the mask is not inherited verbatim -- when a file is created inside a directory
carrying a default ACL the kernel folds the creating mode into it::

    mask &= (requested_mode >> 3)          # fs/posix_acl.c, __posix_acl_create_masq

So the group bits of whatever mode the *writer asked for* decide whether every named-user
entry is effective. Two consequences that cost real debugging time:

* ``tempfile.mkstemp`` requests 0600 by design. Its group bits are zero, so the mask becomes
  ``---`` and every ``u:<role>:r-x`` entry is silently downgraded to ``#effective:---``.
  ``getfacl`` still lists the entry, so the ACL *looks* correct while every read fails with
  EACCES. Publishing such a file through ``os.replace`` preserves that dead mask.
* The umask is **ignored** when a default ACL is present, so ``open(..., "x")`` (which asks
  for 0666) is fine even under ``UMask=0077``. That asymmetry is why some writers in this
  repository were correct by accident and others were not.

:data:`SHARED_READ_MODE` is the mode for anything one uid writes and another must read. It
keeps the mask at ``r--`` -- exactly the access the ACL already granted -- and leaves world
access denied. The containing directories (0700/0750 plus their own ACLs) remain the real
gate; this only stops the mask from cancelling them.

Artifacts already published at 0440/0444 are unaffected: their group read bit keeps the mask
at ``r--`` on its own. Single-uid artifacts (the provider ledger snapshot, the GPU lease) stay
deliberately at 0400/0600 -- nothing outside their owner is meant to read them.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["SHARED_READ_MODE", "chmod_shared"]

#: Owner read/write, group read, no world access.
SHARED_READ_MODE = 0o640


def chmod_shared(path: str | os.PathLike[str]) -> None:
    """Set :data:`SHARED_READ_MODE` on ``path``.

    Call this on the *temporary* file before ``os.replace``, never on the published path
    afterwards: a reader that opens the final name between the rename and the chmod would
    see the dead mask, which is the same bug with a smaller window.
    """
    os.chmod(Path(path), SHARED_READ_MODE)
