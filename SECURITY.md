# Security Policy

## Supported versions

Security fixes are generally applied to the latest released version.

## Reporting a vulnerability

Please do **not** open a public GitHub issue for suspected vulnerabilities.

Report privately to:

- `security@YOUR-DOMAIN.example` (replace with a real address)

Include:

- Affected version / commit
- Reproduction steps
- Impact assessment
- Suggested mitigation (if known)

We will acknowledge receipt as soon as practical and coordinate disclosure and
fix timing with the reporter.

## Scope note

`NonEqSimGPU.jl` is a GPU-only scientific simulation package; most risk surface
is runtime stability and data integrity in simulation workflows rather than
network-facing attack surface.
