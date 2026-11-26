"""Export verified all-particle USD replays from the CRESSim trajectories."""

from pathlib import Path
import json
import numpy as np
from pxr import Usd, UsdGeom, Sdf, Gf, Vt

root = Path(__file__).resolve().parents[2]
cache = root / ".cache/cressim-run/usd-frames"
out = root / "outputs/cressim-cucumber"
rest = np.load(root / "examples/cucumber/2026_0000/mpm.npz")["x"]
plane = float((rest[:, 1].min() + rest[:, 1].max()) / 2)
n = len(rest)


def box(stage, name, center, size, color):
    mesh = UsdGeom.Cube.Define(stage, name)
    mesh.CreateSizeAttr(1.0)
    mesh.CreateDisplayColorAttr([color])
    move = mesh.AddTranslateOp()
    move.Set(Gf.Vec3d(*center))
    mesh.AddScaleOp().Set(Gf.Vec3f(*size))
    return move


def write(mode):
    full = mode == "full"
    native = mode == "native-keyframes"
    main = np.load(
        cache.parent / "refined/native_load.npz"
        if native
        else cache / ("split_load.npz" if full else "half_load.npz")
    )
    ids = main["visual_ids"]
    rows = json.loads(str(main["rows"]))
    if native:
        phases = ["intact", "half_depth", "through_depth", "withdraw", "move_away"]
        rows = [[r for r in rows if r["phase"] == phase][-1] for phase in phases]
        frames = np.stack(
            [
                rest,
                main["half_depth"],
                main["full_depth"],
                main["withdrawn"],
                main["end"],
            ]
        )
        ids = np.arange(n)
    else:
        assert np.array_equal(ids, np.arange(n))
        frames = main["frames"]
    boundary = (
        max(i for i, r in enumerate(rows) if r["phase"] == "through_depth")
        if full
        else len(rows)
    )
    groups = {"Whole": np.ones(n, bool)}
    if full:
        passive = np.load(cache / "split_passive.npz")
        assert np.array_equal(main["side"], passive["side"]) and np.array_equal(
            ~passive["keep"], main["keep"]
        )
        assert json.loads(str(passive["rows"]))[-1]["time_s"] == rows[-1]["time_s"]
        # Until the hand-off both trajectories contain the complete object.
        passive_frames = passive["frames"]
        for i in range(boundary + 1, len(frames)):
            frames[i, passive["keep"]] = passive_frames[i, passive["keep"]]
        groups.update(Piece_001=main["keep"], Piece_002=passive["keep"])
    assert np.isfinite(frames).all()
    dest = out / (mode + "-cut.usdc")
    stage = Usd.Stage.CreateNew(str(dest))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    UsdGeom.Xform.Define(stage, "/World/Fruit")
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(len(rows) - 1)
    stage.SetTimeCodesPerSecond(8)
    stage.SetFramesPerSecond(8)
    stage.GetRootLayer().customLayerData = {
        "cameraSettings": {
            "Perspective": {
                "position": Gf.Vec3d(0.54, 0.08, 0.26),
                "target": Gf.Vec3d(-0.026, -0.01, 0.10),
            }
        }
    }
    clock = world.GetPrim().CreateAttribute(
        "trial:timeSeconds", Sdf.ValueTypeNames.Double
    )
    phase = world.GetPrim().CreateAttribute("trial:phase", Sdf.ValueTypeNames.String)
    blade = box(
        stage,
        "/World/Slicer",
        (-0.025, plane, 0.15),
        (0.16, 0.004, 0.0614),
        (0.50, 0.57, 0.66),
    )
    for i, (lo, hi) in enumerate([(-0.18, plane - 0.012), (plane + 0.012, 0.18)]):
        box(
            stage,
            "/World/Support" + str(i),
            (0, (lo + hi) / 2, 0.018),
            (0.3, hi - lo, 0.026),
            (0.15, 0.20, 0.22),
        )
    colors = {
        "Whole": (0.17, 0.55, 0.31),
        "Piece_001": (0.12, 0.57, 0.33),
        "Piece_002": (0.83, 0.39, 0.13),
    }
    for label, mask in groups.items():
        parent = UsdGeom.Xform.Define(stage, "/World/Fruit/" + label)
        show = parent.CreateVisibilityAttr()
        if full:
            show.Set("inherited" if label == "Whole" else "invisible", 0)
            show.Set("invisible" if label == "Whole" else "inherited", boundary)
        p = UsdGeom.Points.Define(stage, parent.GetPath().AppendChild("Particles"))
        p.CreateWidthsAttr([0.0018])
        p.SetWidthsInterpolation(UsdGeom.Tokens.constant)
        p.CreateDisplayColorAttr([colors[label]])
        p.CreateIdsAttr(Vt.Int64Array.FromNumpy(ids[mask].astype("int64")))
        points = p.CreatePointsAttr()
        extent = p.CreateExtentAttr()
        for i, frame in enumerate(frames):
            if full and ((label == "Whole") != (i < boundary)):
                continue
            subset = np.ascontiguousarray(frame[mask], dtype="float32")
            points.Set(Vt.Vec3fArray.FromNumpy(subset), i)
            extent.Set(
                Vt.Vec3fArray.FromNumpy(np.stack((subset.min(0), subset.max(0)))), i
            )
    for i, row in enumerate(rows):
        clock.Set(row["time_s"], i)
        phase.Set(row["phase"], i)
        blade.Set(Gf.Vec3d(-0.025, plane, row["tip_z_m"] + 0.0307), i)
    stage.GetRootLayer().Save()
    reopened = Usd.Stage.Open(str(dest))
    for i in range(len(rows)):
        active = []
        for g in reopened.GetPrimAtPath("/World/Fruit").GetChildren():
            if UsdGeom.Imageable(g).ComputeVisibility(i) != "invisible":
                p = UsdGeom.Points(g.GetChild("Particles"))
                group_ids = np.asarray(p.GetIdsAttr().Get())
                pos = np.asarray(p.GetPointsAttr().Get(i))
                assert np.array_equal(pos, frames[i, group_ids])
                active.append(group_ids)
        assert np.array_equal(np.sort(np.concatenate(active)), ids)
    print(
        mode,
        len(rows),
        "frames checked, all",
        n,
        "IDs and positions verified",
        flush=True,
    )


if __name__ == "__main__":
    write("half")
    write("full")
    write("native-keyframes")
