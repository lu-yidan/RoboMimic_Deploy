#!/usr/bin/env bash
# Preserve the existing rb -> sr aliases used on G1.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$HERE/deploy/start_tmux_layout.sh" "$@"
