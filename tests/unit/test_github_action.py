"""Static validation of the bundled GitHub composite action
(.github/actions/lint/action.yml + entrypoint.sh).

These are plain text/regex assertions against the YAML and shell files
rather than a parsed-YAML comparison, deliberately: PyYAML is not a
declared project dependency (pyproject.toml has no [tool.*] or
[project.optional-dependencies] entry for it), and adding one for a single
static-file check is not justified per CONTRIBUTING.md's "no new
dependency added without justification" rule.

Regression guards for three bugs that made the published action
non-functional for any consumer but this repository itself:

  1. The action's own step invoked its entrypoint by a path relative to
     the CONSUMER's checkout (`.github/actions/lint/entrypoint.sh`)
     instead of the action's own directory (`${{ github.action_path }}`),
     so `uses: Kunal-Somani/trajlens/.github/actions/lint@v0.4.0` from any
     other repository failed with "No such file or directory".
  2. entrypoint.sh pinned `trajlens==0.3.0` while the action itself (and
     --sarif, which it unconditionally passes) was introduced in v0.4.0 --
     0.3.0 predates the --sarif flag entirely, so every invocation failed
     on an unrecognized option before reaching any lint logic.
  3. action.yml's `outputs:` block had no `value:` mapping to the internal
     step's outputs, so `grade`/`trust-score`/`sarif-file` were always
     empty strings for any consumer regardless of what entrypoint.sh wrote
     to $GITHUB_OUTPUT.
"""

from __future__ import annotations

from pathlib import Path

import trajlens

_ACTION_DIR = Path(__file__).parent.parent.parent / ".github" / "actions" / "lint"


class TestActionYamlInvokesItsOwnEntrypoint:
    def test_run_step_uses_action_path_not_a_relative_path(self) -> None:
        content = (_ACTION_DIR / "action.yml").read_text()
        assert "${{ github.action_path }}/entrypoint.sh" in content
        # The old, broken relative invocation must not have crept back in.
        assert "run: |\n        .github/actions/lint/entrypoint.sh" not in content


class TestActionYamlOutputsAreWired:
    def test_every_declared_output_has_a_value_mapping(self) -> None:
        content = (_ACTION_DIR / "action.yml").read_text()
        for output_name in ("grade", "trust-score", "sarif-file"):
            assert f"value: ${{{{ steps.lint.outputs.{output_name} }}}}" in content, (
                f"output {output_name!r} has no value: mapping to the internal step -- "
                f"it will always be empty for any consumer"
            )

    def test_run_step_has_an_id_for_outputs_to_reference(self) -> None:
        content = (_ACTION_DIR / "action.yml").read_text()
        assert "id: lint" in content


class TestEntrypointVersionPin:
    def test_pinned_trajlens_version_is_not_older_than_this_release(self) -> None:
        """entrypoint.sh must never pin a trajlens version older than the one
        this action ships inside of -- an older pin can predate a CLI flag
        the entrypoint unconditionally passes (e.g. --sarif, added in
        v0.4.0), which fails outright rather than merely being stale."""
        content = (_ACTION_DIR / "entrypoint.sh").read_text()
        import re

        match = re.search(r'pip install "trajlens==([0-9.]+)"', content)
        assert match is not None, "entrypoint.sh must pin an exact trajlens version"
        pinned_version = match.group(1)
        assert pinned_version == trajlens.__version__, (
            f"entrypoint.sh pins trajlens=={pinned_version} but this repo is "
            f"at {trajlens.__version__}; bump the pin in lockstep with releases"
        )

    def test_never_floats_to_latest(self) -> None:
        content = (_ACTION_DIR / "entrypoint.sh").read_text()
        assert "trajlens==" in content
        assert "pip install trajlens\n" not in content
        assert "pip install trajlens " not in content
        assert "--upgrade" not in content
