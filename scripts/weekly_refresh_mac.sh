#!/bin/zsh
# Tuesday refresh (nflverse pbp is final by then): rebuild main pipeline + live state table + live models.
# Retries because the Mac may wake from sleep with no network at 9am (the first failure mode we saw).
cd "$(dirname "$0")/.." || exit 1
for attempt in 1 2 3 4 5 6; do
  if .venv/bin/python main.py --refresh && .venv/bin/python -m live.states \
     && .venv/bin/python -m live.ep && .venv/bin/python -m live.model && .venv/bin/python -m live.dist --fit \
     && .venv/bin/python -m live.totals --fit; then
    echo "refresh ok (attempt $attempt)"; exit 0
  fi
  echo "refresh failed (attempt $attempt), retrying in 5 min"; sleep 300
done
exit 1
