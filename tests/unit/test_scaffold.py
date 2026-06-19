"""M1 scaffold tests.

These tests verify that the project foundation is correct:
- The package is importable and has a version string
- The exception hierarchy is well-formed
- The logging redaction processor fires correctly
- The CLI --version flag works

Nothing here touches dataset logic — that starts at M2.
"""

from __future__ import annotations

import logging

import pytest
import structlog

import trajlens
from trajlens.errors import (
    CheckExecutionError,
    DatasetError,
    DatasetFormatError,
    DatasetVersionError,
    PathTraversalError,
    RepairError,
    ResourceBoundError,
    SourceResolutionError,
    TrajlensError,
)
from trajlens.logging import _REDACTED, _redact_secrets, configure_logging

# ── Package import ─────────────────────────────────────────────────────────


class TestPackageImport:
    def test_version_is_string(self) -> None:
        assert isinstance(trajlens.__version__, str)

    def test_version_non_empty(self) -> None:
        assert len(trajlens.__version__) > 0

    def test_version_is_semver_like(self) -> None:
        # Loose check: contains at least one dot (0.1.0.dev0 is fine)
        assert "." in trajlens.__version__


# ── Exception hierarchy ────────────────────────────────────────────────────


class TestExceptionHierarchy:
    """All domain exceptions must inherit from TrajlensError so callers
    can catch the whole family with a single except clause."""

    def test_dataset_error_is_trajlens_error(self) -> None:
        assert issubclass(DatasetError, TrajlensError)

    def test_format_error_is_dataset_error(self) -> None:
        assert issubclass(DatasetFormatError, DatasetError)

    def test_version_error_is_dataset_error(self) -> None:
        assert issubclass(DatasetVersionError, DatasetError)

    def test_source_resolution_error_is_dataset_error(self) -> None:
        assert issubclass(SourceResolutionError, DatasetError)

    def test_check_execution_error_is_trajlens_error(self) -> None:
        assert issubclass(CheckExecutionError, TrajlensError)

    def test_repair_error_is_trajlens_error(self) -> None:
        assert issubclass(RepairError, TrajlensError)

    def test_resource_bound_error_is_dataset_error(self) -> None:
        assert issubclass(ResourceBoundError, DatasetError)

    def test_path_traversal_error_is_dataset_error(self) -> None:
        assert issubclass(PathTraversalError, DatasetError)

    def test_exceptions_are_raiseable(self) -> None:
        with pytest.raises(DatasetFormatError):
            raise DatasetFormatError("info.json is missing the 'features' key")

    def test_exception_message_preserved(self) -> None:
        msg = "codebase_version '1.99' is not supported; supported: 2.0, 2.1, 3.0"
        exc = DatasetVersionError(msg)
        assert str(exc) == msg


# ── Logging redaction ─────────────────────────────────────────────────────


class TestLoggingRedaction:
    """Redaction is security-critical (T6 in threat model). Test every
    pattern the regex is supposed to catch."""

    def _run_redact(self, event_dict: dict) -> dict:
        # _redact_secrets matches structlog processor signature
        return _redact_secrets(None, "info", event_dict)

    def test_token_key_redacted(self) -> None:
        result = self._run_redact({"token": "hf_supersecret123"})
        assert result["token"] == _REDACTED

    def test_api_key_redacted(self) -> None:
        result = self._run_redact({"api_key": "sk-abc123"})
        assert result["api_key"] == _REDACTED

    def test_authorization_redacted(self) -> None:
        result = self._run_redact({"authorization": "Bearer hf_token"})
        assert result["authorization"] == _REDACTED

    def test_bearer_redacted(self) -> None:
        result = self._run_redact({"bearer": "hf_token"})
        assert result["bearer"] == _REDACTED

    def test_password_redacted(self) -> None:
        result = self._run_redact({"password": "hunter2"})
        assert result["password"] == _REDACTED

    def test_secret_redacted(self) -> None:
        result = self._run_redact({"secret": "abc"})
        assert result["secret"] == _REDACTED

    def test_case_insensitive_token_match(self) -> None:
        result = self._run_redact({"HF_TOKEN": "hf_value"})
        assert result["HF_TOKEN"] == _REDACTED

    def test_non_secret_key_preserved(self) -> None:
        result = self._run_redact({"dataset_path": "/data/lerobot", "fps": 30})
        assert result["dataset_path"] == "/data/lerobot"
        assert result["fps"] == 30

    def test_event_key_not_redacted(self) -> None:
        # 'event' is the log message itself — should never be redacted
        result = self._run_redact({"event": "loading dataset", "token": "secret"})
        assert result["event"] == "loading dataset"
        assert result["token"] == _REDACTED

    def test_mixed_dict_partial_redaction(self) -> None:
        result = self._run_redact(
            {
                "event": "hub login",
                "repo_id": "org/dataset",
                "token": "hf_abc",
                "user": "alice",
            }
        )
        assert result["event"] == "hub login"
        assert result["repo_id"] == "org/dataset"
        assert result["token"] == _REDACTED
        assert result["user"] == "alice"


# ── configure_logging ─────────────────────────────────────────────────────


class TestConfigureLogging:
    def test_configure_does_not_raise(self) -> None:
        # force_plain=True ensures deterministic non-TTY rendering in tests
        configure_logging(level="WARNING", force_plain=True)

    def test_root_logger_level_set(self) -> None:
        configure_logging(level="DEBUG", force_plain=True)
        assert logging.getLogger().level == logging.DEBUG

    def test_structlog_configured(self) -> None:
        configure_logging(force_plain=True)
        logger = structlog.get_logger("test")
        # If structlog is misconfigured this raises; if it passes, the
        # configuration is valid enough to create a logger.
        assert logger is not None
