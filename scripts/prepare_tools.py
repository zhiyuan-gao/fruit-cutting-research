"""Build the blade and board distance fields using the pinned CulinaryCut sampler."""

from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import trimesh
import yaml

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "third_party/EEF-Cutting-Simulation"
sys.path.insert(0, str(UPSTREAM))
from run import _save_pack_npz
from sdf_utils.mesh_sdf import mesh_to_sdf


def prepare_tools():
    folder = ROOT / ".cache/tools"
    folder.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load((ROOT / "configs/simulation.yaml").read_text())
    distance = trimesh.proximity.signed_distance

    def batched(mesh, points):
        return np.concatenate(
            [distance(mesh, points[i : i + 4096]) for i in range(0, len(points), 4096)]
        )

    with (
        TemporaryDirectory(dir=folder) as temp,
        patch("trimesh.proximity.signed_distance", batched),
    ):
        for tool in ("knife", "board"):
            path = folder / (tool + ".npz")
            if path.exists():
                continue
            mesh = UPSTREAM / "assets" / (tool + ".obj")
            settings = config[tool]
            pack = mesh_to_sdf(
                str(mesh), {}, settings["sdf_voxel"], knife_blade=settings.get("blade")
            )
            pending = Path(temp) / path.name
            _save_pack_npz(str(pending), pack)
            pending.replace(path)


if __name__ == "__main__":
    prepare_tools()
