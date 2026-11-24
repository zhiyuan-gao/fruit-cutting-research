"""Rebuild layered fruit templates from the pinned sources and registered anatomy."""

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import trimesh
import yaml

from build_fruit_interiors import build as build_interior
from build_tissue_volumes import (
    ROOT,
    UPSTREAM,
    build as build_gaussian,
    build_banana,
    file_hash,
)
from prepare_stems import apple, peach
from prepare_avocado import prepare_sdf, whole_mesh
from prepare_tools import prepare_tools
from export_fruit_nodes import export
from run import _sdf_cache_key

FRUITS = (
    "apple",
    "avocado",
    "banana",
    "cucumber",
    "melon",
    "orange",
    "peach",
    "strawberry",
)
# The original solver distributes these registered distance grids with its mesh files.
NATIVE = {
    "banana": "49b82aef1ec8d5c5",
    "cucumber": "42350eff172e367f",
    "melon": "39c69c51116522b1",
    "orange": "0d3131804910fe32",
    "strawberry": "14f8e58b712e7bd7",
}


def install_field(source, target):
    # The upstream cache key includes the absolute mesh path and modification time.
    # Publish a complete reference atomically under the key of this checkout.
    with TemporaryDirectory(dir=target.parent) as temp:
        entry = Path(temp) / target.name
        entry.symlink_to(source.resolve(strict=True))
        entry.replace(target)


def prepare(name):
    config = (
        ROOT / "configs/avocado.yaml"
        if name == "avocado"
        else UPSTREAM / "configs" / (name + ".yaml")
    )
    settings = yaml.safe_load(config.read_text())
    cache = ROOT / ".cache/sdf" / hashlib.sha256(config.read_bytes()).hexdigest()[:16]
    cache.mkdir(parents=True, exist_ok=True)
    folder = ROOT / "assets/tissues"
    spec = folder / (name + ".json")
    mesh = (UPSTREAM / settings["cutting_mesh"]["mesh_path"]).resolve()
    geometry = None
    if name in ("apple", "peach"):
        (apple if name == "apple" else peach)()
        mesh = folder / json.loads(spec.read_text())["mesh_variant"]["mesh"]
        geometry = trimesh.load(mesh, force="mesh", skip_materials=True)
        geometry.merge_vertices(merge_tex=True, merge_norm=True)
    elif name == "avocado":
        geometry, _ = whole_mesh(mesh)
    if geometry is None:
        key = _sdf_cache_key(str(mesh), {}, settings["cutting_mesh"]["sdf_voxel"])
        install_field(
            UPSTREAM / "cache_sdf" / f"cuttingmesh.{NATIVE[name]}.npz",
            cache / f"cuttingmesh.{key}.npz",
        )
    else:
        prepare_sdf(geometry, mesh, settings, cache, tissues=(name == "avocado"))
    for tool in ("knife", "board"):
        extra = {"blade": settings["knife"]["blade"]} if tool == "knife" else {}
        source = UPSTREAM / settings[tool]["mesh_path"]
        key = _sdf_cache_key(str(source), {}, settings[tool]["sdf_voxel"], extra)
        install_field(
            ROOT / ".cache/tools" / (tool + ".npz"), cache / f"{tool}.{key}.npz"
        )

    gaussians = ROOT / ".cache/gaussianfluent"
    if name == "banana":
        build_banana()
    elif name in ("orange", "melon"):
        build_gaussian(name, gaussians)
    elif name != "avocado":
        build_interior(name, gaussians)
    template = ROOT / "templates" / name
    # The registered labels are the authoritative research data. Check the reconstruction
    # before publishing the model so registration and physics stay in agreement.
    with (
        np.load(template / "anatomy.npz") as expected,
        np.load(folder / (name + ".npz")) as actual,
    ):
        for field in ("labels", "lobes"):
            np.testing.assert_array_equal(
                actual[field], expected[field], err_msg=f"{name} {field}"
            )
    with TemporaryDirectory(dir=template) as tmp:
        package = Path(tmp) / (name + ".usdz")
        export(config, spec, package)
        checksum = file_hash(package)
        package.replace(template / "base.usdz")
        checksum_file = Path(tmp) / "base.sha256"
        checksum_file.write_text(checksum + "\n")
        checksum_file.replace(template / "base.sha256")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fruit", nargs="+", choices=FRUITS, default=FRUITS)
    args = parser.parse_args()
    prepare_tools()
    for name in args.fruit:
        prepare(name)
