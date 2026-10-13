#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION_NAME="${1:-robomimic}"
DETACH="${2:-}"
command -v tmux >/dev/null
if ! tmux has-session -t "=${SESSION_NAME}" 2>/dev/null; then
  printf -v shell_command 'bash --noprofile --rcfile %q -i' "${REPO_DIR}/tools/tmux_scalebfm.bashrc"
  tmux new-session -d -s "${SESSION_NAME}" -n control -c "${REPO_DIR}" "${shell_command}"
  tmux new-window -d -t "${SESSION_NAME}" -n check -c "${REPO_DIR}" "${shell_command}"
fi
if [[ "${DETACH}" == "--detach" ]]; then
  exit 0
fi
if [[ -n "${TMUX:-}" ]]; then
  tmux switch-client -t "${SESSION_NAME}"
else
  exec tmux attach-session -t "${SESSION_NAME}"
fi
