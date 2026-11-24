"""Check generated assets, or exercise localized knife/body contacts in the actual solver."""

import argparse
import json
import hashlib
import tempfile
import zipfile
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull, cKDTree
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade
from generate import FRUITS, ROOT, components, generate


def assets(reference=False):
    expected = json.loads((ROOT / "records/reference-samples.json").read_text()) if reference else {}
    checked = set()
    results = {}
    for path in sorted((ROOT / "examples").glob("*/*/sample.json")):
        folder = path.parent
        rec = json.loads(path.read_text())
        name = rec["fruit"]
        cfg = json.loads((ROOT / "templates" / name / "template.json").read_text())
        scene = Usd.Stage.Open(str(folder / "fruit.usdz"))
        assert str(UsdGeom.GetStageUpAxis(scene)) == "Z" and UsdGeom.GetStageMetersPerUnit(scene) == 1.0
        textures = []
        for p in scene.Traverse():
            if p.IsA(UsdShade.Shader) and UsdShade.Shader(p).GetIdAttr().Get() == "UsdUVTexture":
                asset = UsdShade.Shader(p).GetInput("file").Get()
                assert asset.resolvedPath, asset
                textures.append(asset.path)
        assert textures
        if name in ("cucumber", "orange"):
            component = "/World/CulinaryCut/Fruit"
            assert not scene.GetPrimAtPath(component + "/Seeds")
            assert not any(p.HasAPI(UsdPhysics.RigidBodyAPI) for p in scene.Traverse())
            if name == "cucumber":
                chamber = UsdGeom.Mesh.Get(scene, component + "/SeedChamber")
                _, _, labels = components(chamber)
                assert len(np.unique(labels)) == 1, "Seed chamber contains disconnected inner surfaces"
        with zipfile.ZipFile(folder / "fruit.usdz") as z:
            for texture in textures:
                assert texture.removeprefix("./") in z.namelist()
        with np.load(folder / "mpm.npz") as data:
            x = data["x"]
            assert np.isfinite(x).all() and len(x) == rec["particle_count"]
            assert np.all(np.isfinite(data["volume_m3"]) & (data["volume_m3"] > 0))
            assert set(data["tissue"]) == {
                i for i, m in enumerate(rec["material_order"], 1) if rec["materials"][m]["mode"] == "soft_mpm"
            }
            dist = []
            tree = cKDTree(x)
            for i, body in enumerate(rec["rigid_bodies"]):
                prim = scene.GetPrimAtPath(body["path"])
                pts = data[f"body_{i}"].astype(float)
                if body["mode"] == "dynamic_rigid":
                    assert prim.HasAPI(UsdPhysics.RigidBodyAPI)
                    mesh = UsdGeom.Mesh.Get(scene, body["path"] + "/Geometry")
                    assert mesh.GetPrim().HasAPI(UsdPhysics.CollisionAPI)
                    visual = np.asarray(mesh.GetPointsAttr().Get()) + np.asarray(
                        UsdGeom.Xformable(prim).GetOrderedXformOps()[0].Get()
                    )
                else:
                    assert prim.HasAPI(UsdPhysics.CollisionAPI) and not prim.HasAPI(UsdPhysics.RigidBodyAPI)
                    visual = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get())
                assert cKDTree(visual).query(pts)[0].max() < 2.0e-7
                hull = ConvexHull(pts)
                close = x[((x >= pts.min(0)) & (x <= pts.max(0))).all(1)]
                for chunk in np.array_split(close, max(1, len(close) // 300)):
                    assert not (
                        (chunk @ hull.equations[:, :3].T + hull.equations[:, 3]).max(1) < -2.0e-7
                    ).any()
                dist.append(float(np.median(tree.query(pts)[0])))
            assert all(d < 3 * rec["spacing_m"] for d in dist), dist
            if name in ("cucumber", "orange"):
                assert not rec["rigid_bodies"] and not ({"Seed", "Seeds"} & rec["materials"].keys())
            if name == "orange":
                assert set(data["segment"]) >= set(range(1, 11))
            if reference and folder.name == "2026_0000":
                baseline = expected[name]
                fingerprint = hashlib.sha256()
                for key in sorted(data.files):
                    fingerprint.update(key.encode() + b"\x00" + data[key].tobytes(order="C"))
                assert fingerprint.hexdigest() == baseline["mpm_arrays_sha256"], f"{name}: reference particle data differ"
                with zipfile.ZipFile(folder / "fruit.usdz") as archive:
                    for image, digest in baseline["textures"].items():
                        assert hashlib.sha256(archive.read(image)).hexdigest() == digest, (name, image)
                assert rec["stage"] == baseline["stage"] and len(rec["rigid_bodies"]) == baseline["rigid_bodies"]
                checked.add(name)
        for m, mat in rec["materials"].items():
            spec = dict(cfg["materials"][m], **cfg["stages"][rec["stage"]]["materials"].get(m, {}))
            for key in spec.keys() - {"mode"}:
                lo, hi = spec[key]["range"]
                assert lo <= mat[key] <= hi
        results[str(folder.relative_to(ROOT))] = dict(
            stage=rec["stage"],
            particles=len(x),
            rigids=len(dist),
            textures=textures,
            max_median_surface_to_soft_m=max(dist, default=0.0),
        )
    assert {k.split("/")[1] for k in results} == set(FRUITS)
    if reference:
        assert checked == set(expected), "Generate all eight 2026_0000 reference samples"
    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp) / "repeat"
        original = ROOT / "examples/banana/2026_0000"
        generate("banana", 2026, 0, dest)
        assert (dest / "sample.json").read_bytes() == (original / "sample.json").read_bytes()
        with np.load(dest / "mpm.npz") as a, np.load(original / "mpm.npz") as b:
            for key in a:
                np.testing.assert_array_equal(a[key], b[key])
        with zipfile.ZipFile(dest / "fruit.usdz") as a, zipfile.ZipFile(original / "fruit.usdz") as b:
            for key in a.namelist():
                assert a.read(key) == b.read(key)
    return dict(samples=results, repeated_banana="identical metadata, particle arrays, USD layer and texture")


def physics(folder, through=False):
    from simulate import FruitSimulation, TO_WORLD

    sim = FruitSimulation(folder)
    try:
        if through:
            if not sim.seeds:
                raise ValueError("This trial requires a seeded fruit")
            centers, _ = sim.seeds.poses()
            target = int(centers[:, 1].argmax())
            top = sim.rest_usd[:, 2].max() + 0.001
            bottom = float(sim.sim.board_top_y[None]) + 0.0001
            tip_start = centers[target] @ TO_WORLD.T
            tip_start[2] = top
            speed = 4.0
            ticks = int(np.ceil((top - bottom) / (speed * sim.sim.dt)))
            sim.command(sim.vertical_cut_pose(tip_start), [0, 0, -speed])
            for start in range(0, ticks, 500):
                sim.step(min(500, ticks - start))
                print(f"{min(start + 500, ticks)}/{ticks} ticks", flush=True)
            result = sim.summary()
            result.update(
                test="skin to board, through initial seed location",
                speed_m_s=speed,
                target_seed=target,
                blade_travel_m=top - (sim.control.position + sim.tip)[2],
            )
            assert (
                min(result["cut_side_particles"]) > 0 and result["soft_knife_accumulated_impulse_Ns"] > 0
            ), result
            assert (
                result["knife_seed_contact_points_over_ticks"] > 0
                and result["max_seed_displacement_m"] > 1.0e-4
            )
        elif sim.pit:
            center = sim.pit.vertices.mean(0) @ TO_WORLD.T
            start = sim.vertical_cut_pose(center) + [0.0, 0.0, 0.12]
            sim.command(start)
            through = sim.vertical_cut_pose(center) - [0.0, 0.0, 0.045]
            sim.command(through)
            stopped = sim.control.position.copy()
            assert sim.pit.blocked and np.linalg.norm(stopped - through) > 0.005
            sim.step(3)
            sim.command(start)
            assert not sim.pit.blocked
            result = sim.summary()
            result["blocked_distance_remaining_m"] = float(np.linalg.norm(stopped - through))
            result["released_on_withdrawal"] = True
        elif sim.seeds:
            seeds = sim.seeds
            pb = seeds.pb
            centers, _ = seeds.poses()
            target = int(centers[:, 1].argmax())
            center = centers[target] @ TO_WORLD.T
            # Locate the first intersection along the blade path before advancing MPM.
            # This is a local loaded-contact test, not a full skin-to-seed slice.
            base = sim.vertical_cut_pose(center)
            hit = 0.0
            clear = 0.035

            def gap(z):
                p = (base + [0, 0, z]) @ TO_WORLD
                pb.resetBasePositionAndOrientation(seeds.knife, p, [0, 0, 0, 1], physicsClientId=seeds.client)
                pts = pb.getClosestPoints(
                    seeds.knife, seeds.bodies[target], 0.05, physicsClientId=seeds.client
                )
                return min((v[8] for v in pts), default=0.05)

            assert gap(hit) <= 0 and gap(clear) > 0
            for _ in range(18):
                value = 0.5 * (clear + hit)
                if gap(value) > 0:
                    clear = value
                else:
                    hit = value
            sim.command(base + [0, 0, clear + 0.00004], [0.0, 0.0, -1.5])
            sim.step(160)
            result = sim.summary()
            delta = seeds.poses()[0] - centers
            result["target_seed"] = target
            result["target_displacement_m"] = float(np.linalg.norm(delta[target]))
            result["target_vs_median_displacement_m"] = float(
                np.linalg.norm(delta[target] - np.median(delta, axis=0))
            )
            assert (
                result["knife_seed_contact_points_over_ticks"] > 0
                and result["accumulated_soft_seed_impulse_Ns"] > 0
            )
            assert result["target_vs_median_displacement_m"] > 0.00003, result
        else:
            center = sim.rest_usd.mean(0)
            near = np.linalg.norm(sim.rest_usd[:, :2] - center[:2], axis=1) < 0.005
            # A curved banana can have its bounding-box center outside its flesh.
            if not near.any():
                nearest = np.argmin(np.linalg.norm(sim.rest_usd - center, axis=1))
                center = sim.rest_usd[nearest].copy()
                near = np.linalg.norm(sim.rest_usd[:, :2] - center[:2], axis=1) < 0.005
            center[2] = sim.rest_usd[near, 2].max() + 0.0005
            sim.command(sim.vertical_cut_pose(center), [0.0, 0.0, -4.0])
            sim.step(int(np.ceil(0.035 / (4.0 * sim.sim.dt))))
            result = sim.summary()
            assert (
                min(result["cut_side_particles"]) > 0 and result["soft_knife_accumulated_impulse_Ns"] > 0
            ), result
        state = sim.state()
        assert set(state["rigid_rest_to_world"]) == {b["path"] for b in sim.record["rigid_bodies"]}
        for transform in state["rigid_rest_to_world"].values():
            np.testing.assert_allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1.0e-6)
        return result
    finally:
        sim.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--physics", type=Path)
    ap.add_argument("--reference", action="store_true", help="Match the eight demo samples against the archived particle and texture hashes")
    ap.add_argument("--out", type=Path)
    ap.add_argument(
        "--through", action="store_true", help="Full stroke from above a seeded fruit to the board"
    )
    args = ap.parse_args()
    result = physics(args.physics, args.through) if args.physics else assets(args.reference)
    text = json.dumps(result, indent=2, allow_nan=False)
    if args.out:
        args.out.write_text(text + "\n")
    print(text)
