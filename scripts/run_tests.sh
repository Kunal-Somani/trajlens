#!/usr/bin/env bash
set -euo pipefail
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 exec pytest -p pytest_cov "$@"
