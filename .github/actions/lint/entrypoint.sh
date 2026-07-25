#!/usr/bin/env bash
set -euo pipefail

# Install trajlens pinned -- never float to @latest
pip install "trajlens==0.3.0" --quiet

# Build args array -- NEVER interpolate INPUT_DATASET_REF into a
# shell string (T1 in 06_SECURITY_AND_THREAT_MODEL.md: this value
# is attacker-controlled in fork PR workflows)
args=("lint" "${INPUT_DATASET_REF}")
[[ "${INPUT_DEEP}" == "true" ]] && args+=("--deep")
sarif_file="${INPUT_SARIF_OUTPUT:-trajlens.sarif}"
args+=("--sarif" "$sarif_file")

# Run lint; capture exit code without letting set -e abort us
trajlens "${args[@]}" || lint_exit=$?
lint_exit="${lint_exit:-0}"

# Map exit codes to GitHub Action outputs
case "$lint_exit" in
  0) grade="PASS"   ; conclusion="success"  ;;
  1) grade="WARN"   ; conclusion="neutral"  ;;
  2) grade="FAIL"   ; conclusion="failure"  ;;
  *) grade="ERROR"  ; conclusion="failure"  ;;
esac

echo "grade=${grade}"           >> "$GITHUB_OUTPUT"
echo "sarif-file=${sarif_file}" >> "$GITHUB_OUTPUT"

# trust-score: parse from --json output on a second pass if exit<=1,
# otherwise emit 0
if [[ "$lint_exit" -le 1 ]]; then
  score=$(trajlens lint "${INPUT_DATASET_REF}" --json 2>/dev/null \
    | python3 -c "import sys,json; \
      d=json.load(sys.stdin); print(d.get('trust_score', 0))" \
    || echo "0")
  echo "trust-score=${score}" >> "$GITHUB_OUTPUT"
else
  echo "trust-score=0" >> "$GITHUB_OUTPUT"
fi

# Propagate failure conclusion
[[ "$conclusion" == "failure" && "${INPUT_FAIL_ON}" == "fail" ]] \
  && exit 2
[[ "$conclusion" != "success" && "${INPUT_FAIL_ON}" == "warn" ]] \
  && exit 1
exit 0
