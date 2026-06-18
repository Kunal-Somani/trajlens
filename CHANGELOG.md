# Changelog

All notable changes to trajlens are documented here.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
Versioning: [Semantic Versioning](https://semver.org/spec/v2.0.0.html)

---

## [Unreleased]

### Added
- Project scaffold: `pyproject.toml`, repo layout, CI pipeline
- `errors.py`: typed exception hierarchy
- `logging.py`: structlog configuration with secret redaction (T6 mitigation)
- `cli.py`: Typer CLI skeleton with `--version`; `lint`/`fix`/`web` stubs
- `SECURITY.md`, `CONTRIBUTING.md`, `LICENSE` (Apache-2.0)
- GitHub Actions: lint, typecheck, test, security, build CI jobs
