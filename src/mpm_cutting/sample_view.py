"""Write sampled solver states to an inspectable OpenUSD stage."""

from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, UsdGeom, Vt
from scipy.spatial import ConvexHull


def points(values):
    return Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(values, dtype=np.float32))


def mesh(stage, path, vertices, faces):
    node = UsdGeom.Mesh.Define(stage, path)
    node.CreatePointsAttr(points(vertices))
    node.CreateFaceVertexCountsAttr(
        Vt.IntArray.FromNumpy(np.full(len(faces), 3, np.int32))
    )
    node.CreateFaceVertexIndicesAttr(
        Vt.IntArray.FromNumpy(np.ascontiguousarray(faces.ravel(), np.int32))
    )
    node.CreateSubdivisionSchemeAttr("none")
    node.CreateDoubleSidedAttr(True)
    return node


class SampleView:
    """Store each recorded state at one timeline frame. simulationTime is in seconds."""

    def __init__(self, stage, env, folder, palette):
        self.stage = stage
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        world = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(world.GetPrim())
        stage.SetStartTimeCode(0)
        stage.SetTimeCodesPerSecond(24)
        self.time = world.GetPrim().CreateAttribute(
            "mpm:simulationTime", Sdf.ValueTypeNames.Double
        )
        self.soft = []
        for i, name in enumerate(env.record["material_order"], 1):
            if env.record["materials"][name]["mode"] != "soft_mpm":
                continue
            node = UsdGeom.Points.Define(stage, "/World/Fruit/" + name)
            node.CreateWidthsAttr([env.record["spacing_m"]])
            node.SetWidthsInterpolation("constant")
            node.CreateDisplayColorAttr([Gf.Vec3f(*palette[name])])
            self.soft.append(
                (node.CreatePointsAttr(), np.flatnonzero(env.tissue_ids == i))
            )
        self.rigid = {}
        with np.load(Path(folder) / "mpm.npz") as data:
            for i, body in enumerate(env.record["rigid_bodies"]):
                vertices = data[f"body_{i}"]
                path = body["path"]
                node = mesh(
                    stage,
                    "/World/Fruit/" + path.rsplit("/", 1)[1],
                    vertices,
                    ConvexHull(vertices).simplices,
                )
                node.CreateDisplayColorAttr([Gf.Vec3f(*palette[body["material"]])])
                self.rigid[path] = node.AddTransformOp()
        blade = mesh(
            stage, "/World/Knife", env.knife_mesh.vertices, env.knife_mesh.faces
        )
        blade.CreateDisplayColorAttr([Gf.Vec3f(0.48, 0.51, 0.53)])
        self.translation = blade.AddTranslateOp()
        self.orientation = blade.AddOrientOp()
        board = UsdGeom.Cube.Define(stage, "/World/Board")
        board.CreateSizeAttr(1.0)
        board.AddTranslateOp().Set(
            Gf.Vec3d(0, 0, float(env.sim.board_top_y[None]) - 0.005)
        )
        board.AddScaleOp().Set(Gf.Vec3f(0.55, 0.4, 0.01))
        board.CreateDisplayColorAttr([Gf.Vec3f(0.35, 0.22, 0.13)])
        self.frame = 0

    def update(self, state):
        frame = self.frame
        self.time.Set(float(state["time_s"]), frame)
        for attr, indices in self.soft:
            attr.Set(points(state["positions"][indices]), frame)
        for path, matrix in state["rigid_rest_to_world"].items():
            self.rigid[path].Set(Gf.Matrix4d(*matrix.T.ravel().tolist()), frame)
        knife = state["knife"]
        self.translation.Set(Gf.Vec3d(*knife["position"].tolist()), frame)
        q = knife["quaternion"].tolist()
        self.orientation.Set(Gf.Quatf(q[0], Gf.Vec3f(*q[1:])), frame)
        self.stage.SetEndTimeCode(frame)
        self.frame += 1
