# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |
| 0.0.x   | :white_check_mark: (placeholder release; not for production use) |

We support the latest released minor version. Older versions receive security
fixes only when there is no safe upgrade path.

## Reporting a Vulnerability

Please report vulnerabilities privately to **security@trulayer.ai** or via the
[GitHub Security Advisories](https://github.com/trulayer/client-python/security/advisories/new)
flow on this repository.

- Include a clear description, reproduction steps, and the version affected.
- Do **not** open a public GitHub issue for security reports.
- We aim to acknowledge within 5 business days. This is a best-effort SLA;
  formal SLAs will be published when the SDK reaches `1.0.0`.
- A PGP key for encrypted disclosures will be published here once available.

## Disclosure

We will coordinate disclosure timing with the reporter. Please give us a
reasonable window to ship a fix before publishing details.

## Release signing & supply-chain integrity

Every `trulayer` release published to PyPI is:

1. **Built in GitHub Actions** from a tagged commit in this repository, using
   [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/) (no
   long-lived API token).
2. **Signed with [Sigstore](https://www.sigstore.dev/)** using the workflow's
   short-lived OIDC identity. There are no long-lived signing keys.
3. **Logged in the [Rekor](https://docs.sigstore.dev/logging/overview/)
   transparency log** — every signature is publicly auditable at
   <https://search.sigstore.dev/>.
4. **Accompanied by a [CycloneDX](https://cyclonedx.org/) SBOM** attached to
   the corresponding GitHub Release, listing every resolved dependency.

### Verifying a release with `sigstore`

You can cryptographically verify that a wheel or sdist you downloaded from
PyPI was produced by this repository's release workflow at a specific tag.

```bash
# 1. Install the sigstore CLI
pip install sigstore

# 2. Download the artifact and its .sigstore bundle from the GitHub Release
#    e.g. https://github.com/trulayer/client-python/releases/tag/v0.1.0
VERSION=0.1.0
curl -LO "https://github.com/trulayer/client-python/releases/download/v${VERSION}/trulayer-${VERSION}-py3-none-any.whl"
curl -LO "https://github.com/trulayer/client-python/releases/download/v${VERSION}/trulayer-${VERSION}-py3-none-any.whl.sigstore"

# 3. Verify the wheel was signed by this repo's release.yml at the matching tag
sigstore verify identity \
  --bundle "trulayer-${VERSION}-py3-none-any.whl.sigstore" \
  --cert-identity "https://github.com/trulayer/client-python/.github/workflows/release.yml@refs/tags/v${VERSION}" \
  --cert-oidc-issuer "https://token.actions.githubusercontent.com" \
  "trulayer-${VERSION}-py3-none-any.whl"
```

A successful run prints `OK` and exits 0. Any tampering, identity mismatch,
or missing Rekor entry will fail the verification.

### Finding the Rekor transparency log entry

Every signature is a public Rekor entry. Look it up at
<https://search.sigstore.dev/> by searching the artifact's SHA-256 digest,
or use the CLI:

```bash
sigstore verify identity --offline=false \
  --bundle "trulayer-${VERSION}-py3-none-any.whl.sigstore" \
  ... # see above
```

### SBOM

The CycloneDX JSON SBOM is attached to each GitHub Release as
`trulayer-<version>-sbom.cdx.json`. It captures the resolved dependency graph
at build time and is itself Sigstore-signed.
