#!/usr/bin/env bash
# Syncs the TruLayer OpenAPI spec to tests/fixtures/ for contract testing.
# Usage: ./scripts/sync_openapi.sh
#
# Downloads the published spec from the TruLayer API docs:
#   curl -fSL https://api.trulayer.ai/openapi.yaml -o tests/fixtures/openapi.yaml

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEST="$REPO_ROOT/tests/fixtures/openapi.yaml"
SPEC_URL="${TRULAYER_OPENAPI_URL:-https://api.trulayer.ai/openapi.yaml}"

echo "Fetching OpenAPI spec from $SPEC_URL"
curl -fSL "$SPEC_URL" -o "$DEST"
echo "Synced: $SPEC_URL -> $DEST"
