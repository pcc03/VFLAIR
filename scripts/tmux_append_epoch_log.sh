#!/usr/bin/env bash
# Append-only filter for tmux pipe-pane: keep validation + epoch summary lines.
# Usage (from any shell, while training runs in tmux session 1):
#   tmux pipe-pane -t 1 -o "bash /home/peng326/VFLAIR/scripts/tmux_append_epoch_log.sh"
# Stop logging (training keeps running):
#   tmux pipe-pane -t 1

set -euo pipefail

LOG_DIR="${VFLAIR_EPOCH_LOG_DIR:-/home/peng326/VFLAIR/logs}"
LOG_FILE="${VFLAIR_EPOCH_LOG_FILE:-${LOG_DIR}/tmux_session1_epoch.log}"

mkdir -p "$LOG_DIR"

# Line-buffered: flush each line so the log updates live without waiting for a huge buffer.
stdbuf -oL grep --line-buffered -E 'validate and test|Epoch [0-9]+%[[:space:]]+train_loss:' >>"$LOG_FILE"
