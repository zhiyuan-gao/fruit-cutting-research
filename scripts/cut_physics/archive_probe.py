"""Positive control for the knife-crossing detector, using the archived Isaac timeline."""

import json

import numpy as np
from pxr import Usd, UsdGeom
import warp as wp

from native_probe import audit, ROOT


def particle_and_knife(stage, frame, wanted):
    cache = UsdGeom.XformCache(frame)

    def points(prim):
        m = np.asarray(cache.GetLocalToWorldTransform(prim))
        return np.asarray(prim.GetAttribute("points").Get(frame)) @ m[:3, :3] + m[3, :3]

    knife = stage.GetPrimAtPath("/World/Knife")
    for group in stage.GetPrimAtPath("/World/Fruit").GetChildren():
        if UsdGeom.Imageable(group).ComputeVisibility(frame) == "invisible":
            continue
        for prim in group.GetChildren():
            ids = prim.GetAttribute("ids")
            if ids:
                idx = np.flatnonzero(np.asarray(ids.Get(frame)) == wanted)
                if len(idx):
                    return points(prim)[idx[0]], points(knife)
    raise ValueError(f"Particle {wanted} missing from frame {frame}")


def main():
    stage = Usd.Stage.Open(str(ROOT / "outputs/cucumber-piece-trial/cucumber.usdc"))
    knife = stage.GetPrimAtPath("/World/Knife")
    faces = np.asarray(knife.GetAttribute("faceVertexIndices").Get())
    samples = [particle_and_knife(stage, f, 71186) for f in (4, 5, 6)]
    p = np.array([v[0] for v in samples])[:, None, :]
    v = np.array([v[1] for v in samples])
    translations = (v - v[0]).mean(1)
    if (
        not np.allclose(v - v[0], translations[:, None, :], atol=1.0e-8)
        or abs(translations[:, :2]).max() > 1.0e-8
    ):
        raise ValueError("Expected vertical knife translation in this archive")
    mesh = wp.Mesh(
        wp.array(v[0], dtype=wp.vec3, device="cuda"),
        wp.array(faces, dtype=int, device="cuda"),
    )
    entries, inside = [wp.zeros(2, dtype=int, device="cuda") for _ in range(2)]
    wp.launch(
        audit,
        (2, 1),
        [
            mesh.id,
            wp.array(p, dtype=wp.vec3, device="cuda"),
            wp.array(translations[:, 2], dtype=float, device="cuda"),
            entries,
            inside,
        ],
        device="cuda",
    )
    report = dict(
        particle=71186,
        frames=[4, 5, 6],
        entries=entries.numpy().tolist(),
        inside_at_end=inside.numpy().tolist(),
        motion="linear interpolation of archived frames",
    )
    print(report)
    if report["entries"] != [1, 0] or report["inside_at_end"] != [1, 0]:
        raise RuntimeError("Archive positive control failed")
    path = ROOT / "outputs/cut-physics/archive-check.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
