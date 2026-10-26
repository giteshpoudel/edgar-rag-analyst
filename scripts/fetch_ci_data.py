"""Download the frozen data snapshot (release ci-data-v1), verify it and unpack it.

Used when building the demo container. The expected SHA-256 is committed in
evals/ci-data.sha256, so a changed or corrupted archive fails the build instead of serving
different data. Standard library only (the slim image has no gh or shasum).

    python scripts/fetch_ci_data.py --dest /app     # creates /app/data/...
"""

import argparse
import hashlib
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

URL = "https://github.com/giteshpoudel/edgar-rag-analyst/releases/download/ci-data-v1/edgar-ci-data.tar.gz"
CHECKSUM_FILE = Path(__file__).resolve().parent.parent / "evals" / "ci-data.sha256"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dest", default=".", help="folder to unpack into (gets a data/ subfolder)")
    args = parser.parse_args()

    expected = CHECKSUM_FILE.read_text().split()[0]
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "edgar-ci-data.tar.gz"
        urllib.request.urlretrieve(URL, archive)
        actual = hashlib.sha256(archive.read_bytes()).hexdigest()
        if actual != expected:
            print(f"checksum mismatch: got {actual}, expected {expected}", file=sys.stderr)
            return 1
        with tarfile.open(archive) as tar:
            tar.extractall(args.dest, filter="data")  # "data" refuses absolute paths and ../
    print(f"ci data ok ({expected[:12]}...) -> {Path(args.dest).resolve() / 'data'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
