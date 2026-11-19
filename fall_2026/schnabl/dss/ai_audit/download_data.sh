#!/usr/bin/env bash
# Download FRED target rates. The Python loader downloads the New York Fed reports.
set -euo pipefail
cd -- "$(dirname -- "$0")"
if [[ ! -s fedfunds.csv ]]; then
    curl -fL --retry 3 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=DFEDTAR,DFEDTARU,DFEDTARL&cosd=1992-01-01&coed=2013-12-31' -o fedfunds.csv
fi
