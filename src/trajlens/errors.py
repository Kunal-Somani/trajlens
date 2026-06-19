"""Typed exception hierarchy for trajlens.

Every exception includes a human-readable message that states what went wrong
and, where possible, what the user can do about it. Bare Exception is never raised.
"""


class TrajlensError(Exception):
    """Base class for all trajlens errors."""


class DatasetError(TrajlensError):
    """Base class for all dataset-related errors."""


class DatasetFormatError(DatasetError):
    """Dataset structure or content violates the LeRobotDataset format spec.

    Raised when metadata is unparseable, a required file is absent, or a
    structural invariant is violated during loading (not during checks — check
    violations produce CheckResults, not exceptions).
    """


class DatasetVersionError(DatasetError):
    """Dataset format version is unrecognised or unsupported.

    Include the detected version string and the supported range in the message.
    """


class SourceResolutionError(DatasetError):
    """A dataset reference (local path or Hub repo id) could not be resolved.

    Raised when the path does not exist, the Hub repo is not found, or
    access is denied.
    """


class CheckExecutionError(TrajlensError):
    """A check crashed during execution.

    This is an internal error — the check engine catches it and converts it to
    an ERROR CheckResult. It is never propagated to the user as an unhandled
    exception.
    """


class RepairError(TrajlensError):
    """A repair operation failed or was aborted.

    Raised when a fixer encounters an unrecoverable state. The original dataset
    is never modified when this is raised (copy-on-write guarantee per ADR-004).
    """


class ResourceBoundError(DatasetError):
    """A dataset-declared count exceeds the hard ceiling for safe processing.

    Raised instead of attempting to iterate over a maliciously or accidentally
    huge declared size (T2 in the threat model). Includes the declared value
    and the ceiling in the message.
    """


class PathTraversalError(DatasetError):
    """A path derived from dataset metadata escapes the dataset root.

    Raised by safe_join() when a resolved path would leave the dataset
    directory (T1 in the threat model).
    """
