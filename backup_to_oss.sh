#!/usr/bin/env bash
# Run from any directory; all options are forwarded to the Python implementation.
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${PYTHON:-python3}" "$PROJECT_DIR/scripts/backup_to_oss.py" "$@"
