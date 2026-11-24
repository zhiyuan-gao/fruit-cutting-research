"""Download the exact public inputs required to rebuild the eight fruit models."""

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
from tempfile import NamedTemporaryFile

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open("rb") as stream:
        result = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def fetch(entry):
    path = ROOT / entry["path"]
    if path.is_file() and digest(path) == entry["sha256"]:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temp = Path(stream.name)
    try:
        subprocess.run(
            [
                "curl",
                "--fail",
                "--location",
                "--silent",
                "--show-error",
                "--retry",
                "3",
                "--user-agent",
                "FruitCuttingResearch/1.0 (research asset preparation)",
                "--output",
                str(temp),
                entry["url"],
            ],
            check=True,
        )
        if digest(temp) != entry["sha256"]:
            raise ValueError(
                f"Downloaded content differs from the pinned SHA-256 for {path.name}"
            )
        os.replace(temp, path)
        print(entry["path"], flush=True)
    finally:
        temp.unlink(missing_ok=True)


if __name__ == "__main__":
    entries = json.loads((ROOT / "sources.lock.json").read_text())
    with ThreadPoolExecutor(max_workers=5) as pool:
        list(pool.map(fetch, entries))
    print(f"Verified {len(entries)} upstream files.")
