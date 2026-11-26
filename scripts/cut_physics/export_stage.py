"""Export measured coupon states. Every visible USD point keeps its original particle ID."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, Vt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/mpm_cutting"))
from pieces import SingleCutPieces  # noqa: E402
from reference_cut import knife_geometry  # noqa: E402
from sample_view import mesh, points  # noqa: E402


def export(folder, output):
    config = json.loads((folder / "result.json").read_text())
    timeline = np.load(folder / "timeline.npz")
    manipulation = np.load(folder / "manipulation.npz")
    x = np.concatenate([timeline["x"], manipulation["x"]])
    side = np.concatenate([timeline["side"], manipulation["side"]])
    heights = np.r_[timeline["knife_z"], manipulation["knife_z"]]
    times = np.r_[timeline["time_s"], timeline["time_s"][-1] + manipulation["time_s"]]
    tracker = SingleCutPieces(x[0], config["spacing_m"])
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(len(x) - 1)
    stage.SetTimeCodesPerSecond(24)
    time_s = world.GetPrim().CreateAttribute(
        "mpm:timeSeconds", Sdf.ValueTypeNames.Double
    )
    verts, faces, _ = knife_geometry(ROOT)
    verts[:, 1] += config["lateral_mm"] * 0.001
    knife = mesh(stage, "/World/Knife", verts, faces)
    knife.CreateDisplayColorAttr([(0.53, 0.59, 0.62)])
    pose = knife.AddTranslateOp()
    prims = []
    for name, color in [
        ("Whole", (0.56, 0.78, 0.29)),
        ("Piece_001", (0.27, 0.57, 0.27)),
        ("Piece_002", (0.73, 0.82, 0.35)),
    ]:
        group = UsdGeom.Xform.Define(stage, "/World/Coupon/" + name)
        node = UsdGeom.Points.Define(stage, group.GetPath().AppendChild("Material"))
        node.CreateWidthsAttr([config["spacing_m"] * 0.88])
        node.SetWidthsInterpolation("constant")
        node.CreateDisplayColorAttr([color])
        prims.append((group, node))
    for f, (pos, tags, height, clock) in enumerate(zip(x, side, heights, times)):
        labels = tracker.update(np.choose(tags, [0, -1, 1]))
        pose.Set(Gf.Vec3d(0.0, 0.0, float(height)), f)
        time_s.Set(float(clock), f)
        for part, (group, node) in enumerate(prims):
            ids = np.flatnonzero(labels == part)
            group.GetVisibilityAttr().Set("inherited" if len(ids) else "invisible", f)
            if len(ids):
                node.GetPointsAttr().Set(points(pos[ids]), f)
                node.GetIdsAttr().Set(Vt.Int64Array.FromNumpy(ids.astype("i8")), f)
    stage.GetRootLayer().customLayerData = {
        "cameraSettings": {
            "Perspective": {
                "position": Gf.Vec3d(0.025, -0.046, 0.022),
                "target": Gf.Vec3d(0.0, 0.0, 0.004),
            }
        }
    }
    if not stage.GetRootLayer().Export(str(output)):
        raise OSError(output)
    check = Usd.Stage.Open(str(output))
    for f in range(len(x)):
        visible = []
        for group, node in prims:
            g = UsdGeom.Imageable(check.GetPrimAtPath(group.GetPath()))
            if g.ComputeVisibility(f) != "invisible":
                p = UsdGeom.Points(check.GetPrimAtPath(node.GetPath()))
                ids = np.asarray(p.GetIdsAttr().Get(f))
                assert np.array_equal(np.asarray(p.GetPointsAttr().Get(f)), x[f, ids])
                visible.extend(ids)
        assert np.array_equal(np.sort(visible), np.arange(x.shape[1]))
    print(
        f"{output}: verified {len(x)} frames and {x.shape[1]} unique particle IDs per frame"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    export(args.folder, args.output)
