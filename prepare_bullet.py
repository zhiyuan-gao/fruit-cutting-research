"""Install the single optional rigid solver extension for the bundled Linux CPython 3.11 runtime."""

from pathlib import Path
import hashlib
import tempfile
import urllib.request
import json
import zipfile

ROOT = Path(__file__).resolve().parent
WHEEL = "pybullet-3.2.7-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
DIGEST = "5d858abf7c12a21ef3ed351cf81dac1b5e486ec25c9ab59bfc84beb89034edff"
if __name__ == "__main__":
    cache = ROOT / ".cache/wheels"
    cache.mkdir(parents=True, exist_ok=True)
    dest = cache / WHEEL
    if not dest.exists():
        with urllib.request.urlopen("https://pypi.org/pypi/pybullet/3.2.7/json") as response:
            url = next(item["url"] for item in json.load(response)["urls"] if item["filename"] == WHEEL)
        with tempfile.NamedTemporaryFile(dir=cache) as temp:
            urllib.request.urlretrieve(url, temp.name)
            data = Path(temp.name).read_bytes()
            if hashlib.sha256(data).hexdigest() != DIGEST:
                raise ValueError("Wheel hash mismatch")
            dest.write_bytes(data)
    if hashlib.sha256(dest.read_bytes()).hexdigest() != DIGEST:
        raise ValueError("Wheel hash mismatch")
    with zipfile.ZipFile(dest) as archive:
        for item in archive.namelist():
            if (item.startswith("pybullet.cpython") and item.endswith(".so")) or (
                ".dist-info/" in item and "LICENSE" in item
            ):
                archive.extract(item, ROOT / ".cache/bullet")
