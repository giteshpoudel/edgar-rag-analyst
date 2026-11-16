#!/usr/bin/env bash
# Pack the frozen data snapshot the CI gate runs on (W6): the LanceDB index, the processed
# chunks, and the cached XBRL company facts of each indexed company. No raw 10-K HTML.
#
#   scripts/ci_data.sh pack  OUT.tar.gz      # build the archive and print its sha256
#   scripts/ci_data.sh check FILE.tar.gz     # verify it against evals/ci-data.sha256
#   scripts/ci_data.sh pack-v3  OUT.tar.gz   # v3 add-on (ci-data-v2): the 5-year LanceDB table
#   scripts/ci_data.sh check-v3 FILE.tar.gz  #   and the claims database; extracted on top of v1
#
# CI downloads the archive from the GitHub release named in .github/workflows/ci.yml and
# refuses to run if the checksum doesn't match, so the gate always scores the same data.
set -euo pipefail
cmd=${1:?pack or check}; file=${2:?archive path}

if [ "$cmd" = pack ]; then
  tickers=$(ls data/processed)
  files="data/lancedb data/processed data/raw/company_tickers.json"
  for t in $tickers; do files="$files data/raw/$t/companyfacts.json"; done
  # Fixed order, owner and timestamps, so the same data always gives the same checksum.
  tar --sort=name --owner=0 --group=0 --mtime='2026-01-01' -czf "$file" $files 2>/dev/null \
    || COPYFILE_DISABLE=1 tar -czf "$file" $files   # macOS tar lacks --sort; checksum then varies
  shasum -a 256 "$file" | cut -d' ' -f1
elif [ "$cmd" = pack-v3 ]; then
  files="data/lancedb/chunks_multi.lance data/v3/claims.sqlite"
  tar --sort=name --owner=0 --group=0 --mtime='2026-01-01' -czf "$file" $files 2>/dev/null \
    || COPYFILE_DISABLE=1 tar -czf "$file" $files
  shasum -a 256 "$file" | cut -d' ' -f1
elif [ "$cmd" = check-v3 ]; then
  expected=$(cut -d' ' -f1 evals/ci-data-v2.sha256)
  actual=$(shasum -a 256 "$file" | cut -d' ' -f1)
  [ "$expected" = "$actual" ] || { echo "ci data v2 checksum mismatch: $actual != $expected" >&2; exit 1; }
  echo "ci data v2 checksum ok"
elif [ "$cmd" = check ]; then
  expected=$(cut -d' ' -f1 evals/ci-data.sha256)
  actual=$(shasum -a 256 "$file" | cut -d' ' -f1)
  [ "$expected" = "$actual" ] || { echo "ci data checksum mismatch: $actual != $expected" >&2; exit 1; }
  echo "ci data checksum ok"
fi
