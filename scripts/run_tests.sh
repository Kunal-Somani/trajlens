#!/usr/bin/env bash
# Wrapper to run pytest in a way that avoids the ROS2 launch_testing plugin
# conflict (PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 suppresses all entrypoint-based
# plugins, then -p pytest_cov re-enables just coverage explicitly).
#
# Use this instead of calling pytest directly.  Works with or without having
# first run 'source .venv/bin/activate'; if the venv's pytest is on PATH it
# will be used, otherwise we fall back to the venv-relative path.
#
# Why PYTEST_DISABLE_PLUGIN_AUTOLOAD=1: the ROS2 launch_testing package
# registers itself as a pytest11 entry point system-wide and it imports
# 'lark', which is not installed in this project's venv.  Disabling all
# autoloaded plugins and explicitly re-enabling pytest_cov is the upstream-
# recommended fix for this class of conflict.  See the commit that introduced
# this script for the full investigation (commit 229166a).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTEST="${SCRIPT_DIR}/../.venv/bin/pytest"

if command -v pytest &>/dev/null; then
    PYTEST_BIN="pytest"
elif [[ -x "${VENV_PYTEST}" ]]; then
    PYTEST_BIN="${VENV_PYTEST}"
else
    echo "ERROR: pytest not found on PATH and not at ${VENV_PYTEST}" >&2
    echo "Run 'source .venv/bin/activate' or 'pip install -e .[dev]' first." >&2
    exit 1
fi

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 exec "${PYTEST_BIN}" -p pytest_cov "$@"
