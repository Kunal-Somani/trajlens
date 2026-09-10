"""Unit tests for repair/protocol.py's shared replace_output_dir() helper.

This is the single sanctioned guard every Fixer.apply() calls before
shutil.copytree()-ing into its output directory. Regression guard for the
data-loss bug it replaces: every fixer used to call
``shutil.rmtree(output_path)`` unconditionally whenever the path already
existed, so `trajlens fix ds --apply --out ~/datasets` would silently
delete the entire contents of ~/datasets if that directory happened to
already exist.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from trajlens.errors import RepairError
from trajlens.repair.protocol import replace_output_dir


class TestReplaceOutputDir:
    def test_nonexistent_path_is_a_noop(self, tmp_path: Path) -> None:
        """A path that doesn't exist yet is left alone -- nothing to remove."""
        target = tmp_path / "does_not_exist_yet"
        replace_output_dir(target)  # must not raise
        assert not target.exists()

    def test_empty_existing_directory_is_removed(self, tmp_path: Path) -> None:
        """An empty pre-existing directory is removed -- nothing is lost."""
        target = tmp_path / "empty_dir"
        target.mkdir()
        replace_output_dir(target)
        assert not target.exists()

    def test_non_empty_existing_directory_is_refused(self, tmp_path: Path) -> None:
        """A non-empty pre-existing directory is refused, not deleted."""
        target = tmp_path / "populated_dir"
        target.mkdir()
        (target / "important_file.txt").write_text("do not delete me")

        with pytest.raises(RepairError, match="already exists and is not empty"):
            replace_output_dir(target)

        # The refusal must not have touched the directory's contents.
        assert (target / "important_file.txt").is_file()
        assert (target / "important_file.txt").read_text() == "do not delete me"

    def test_non_empty_directory_with_only_a_subdirectory_is_refused(self, tmp_path: Path) -> None:
        """Non-emptiness via a subdirectory (not just a file) is also refused."""
        target = tmp_path / "populated_dir"
        target.mkdir()
        (target / "subdir").mkdir()

        with pytest.raises(RepairError, match="already exists and is not empty"):
            replace_output_dir(target)

        assert (target / "subdir").is_dir()
