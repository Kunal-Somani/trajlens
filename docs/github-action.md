# GitHub Action

trajlens publishes a GitHub Action that runs `trajlens lint` in CI and
surfaces findings as SARIF, so they annotate pull request diffs in
GitHub's code-scanning UI. The action is read-only: it never runs
`trajlens fix --apply` against your dataset.

## Usage

```yaml
name: trajlens lint

on: [pull_request]

jobs:
  lint:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      security-events: write
    steps:
      - uses: actions/checkout@v4
      - name: Run trajlens
        id: trajlens
        uses: Kunal-Somani/trajlens/.github/actions/lint@v0.4.0
        with:
          dataset-ref: lerobot/pusht
          deep: false
          fail-on: fail
          sarif-output: trajlens.sarif
      - name: Upload SARIF
        if: always()
        uses: github/codeql-action/upload-sarif@v3
        with:
          sarif_file: trajlens.sarif
```

## Inputs

| Name | Required | Default | Description |
|---|---|---|---|
| `dataset-ref` | yes | — | Local path or Hugging Face Hub repo id (`org/name`) to lint. |
| `deep` | no | `false` | Full video decode instead of spot-check. Slower; catches more. |
| `fail-on` | no | `fail` | `fail` or `warn`. Determines which severities fail the job (see exit codes below). |
| `sarif-output` | no | `trajlens.sarif` | Path to write the SARIF 2.1.0 report to. |

## Outputs

| Name | Description |
|---|---|
| `grade` | The overall lint grade for the dataset. |
| `trust-score` | The numeric trust score (0-100). |
| `sarif-file` | Path to the written SARIF report (same as `sarif-output`). |

## Exit-code semantics

`trajlens lint` maps its worst finding severity to a process exit code,
which the action maps to its conclusion:

| Worst severity | Exit code | Action conclusion |
|---|---|---|
| clean (no WARN/FAIL/ERROR) | `0` | success |
| WARN | `1` | neutral |
| FAIL or ERROR | `2` | failure |

These exit codes are a stable contract and will not change without a
major version bump.

## Security notes

- The action is read-only: it only ever runs `trajlens lint`, never
  `trajlens fix --apply`.
- `dataset-ref` is passed to `trajlens` as a positional argument via an
  array, never interpolated into a shell string, since it may be
  attacker-controlled on PR-triggered workflows against forked repos.
- The action installs `trajlens` pinned to a specific released version,
  not `@latest`.

