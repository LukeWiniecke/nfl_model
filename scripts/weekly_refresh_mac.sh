#!/bin/zsh
# Tuesday refresh (nflverse pbp is final by then): rebuild main pipeline + live state table.
cd "$(dirname "$0")/.." || exit 1
.venv/bin/python main.py --refresh
.venv/bin/python -m live.states
