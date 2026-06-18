# Security Policy

## Supported versions

| Version | Supported |
|---------|-----------|
| latest  | ✅        |

trajlens is pre-v1.0. Security fixes are applied to the latest release only.

## Reporting a vulnerability

**Do not open a public GitHub issue for security vulnerabilities.**

Report security issues by email to: **security@trajlens.dev**
(Replace with your actual security contact before publishing.)

Include:
- Description of the vulnerability and its impact
- Steps to reproduce, or a minimal proof-of-concept dataset/input
- The version of trajlens you are using

Expected response window: **72 hours** for acknowledgement, **14 days** for a fix or
mitigation plan on confirmed issues.

## Disclosure policy

We follow responsible disclosure. Once a fix is released, we will publish a security
advisory describing the vulnerability, its impact, and the fix. We will credit the
reporter unless they prefer to remain anonymous.

## In scope

- Path traversal via dataset metadata (T1)
- Decompression / allocation bombs via declared dataset sizes (T2)
- Deserialization vulnerabilities in metadata parsing (T3)
- Malformed media crashing or hanging the process (T5)
- Hub token or secret leakage in logs, reports, or error messages (T6)
- Any bug in the `fix` command that corrupts a dataset it claimed to repair (T9)
- Web dashboard vulnerabilities when the dashboard is available (T10)

## Out of scope

- Denial-of-service requiring a compromised host
- Social engineering
- Issues in dependencies (report those upstream; we will update our dependency)
