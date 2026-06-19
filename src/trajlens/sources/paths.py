"""Containment-checked path construction (T1 in the threat model).

safe_join() is the only sanctioned way to build a filesystem path from parts
that originate in dataset metadata (path templates, chunk/file indices,
camera keys). Direct os.path.join or Path / "part" on untrusted parts is
banned project-wide — see 06_SECURITY_AND_THREAT_MODEL.md T1.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath

from trajlens.errors import PathTraversalError


def safe_join(root: Path, *parts: str) -> Path:
    """Join *parts* onto *root*, rejecting any result that escapes *root*.

    Each part is split on '/' and rebuilt segment-by-segment so that an
    absolute-looking part (e.g. '/etc/passwd') cannot reset the join to
    outside root, and a '..' segment is rejected outright rather than
    silently resolved. Raises PathTraversalError on any escape attempt.
    """
    base = root.resolve()
    candidate = base
    for part in parts:
        for segment in PurePosixPath(part).parts:
            if segment in ("", ".", "/"):
                continue
            if segment == "..":
                raise PathTraversalError(
                    f"path part {part!r} contains a '..' segment, "
                    f"which is not allowed in dataset-derived paths"
                )
            candidate = candidate / segment

    resolved = candidate.resolve()
    try:
        resolved.relative_to(base)
    except ValueError:
        raise PathTraversalError(f"resolved path {resolved} escapes dataset root {base}") from None
    return resolved
