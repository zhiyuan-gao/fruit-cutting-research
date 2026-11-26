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
        self.env = env
        self.palette = palette
        self.groups = {}
        self.frame = 0
        self.masses = env.sim.particles.mass.to_numpy() if env.pieces is not None else None
        if self.masses is not None:
            self.piece_count = world.GetPrim().CreateAttribute("mpm:pieceCount", Sdf.ValueTypeNames.Int)
        else:
            self.add_group(0, np.arange(len(env.tissue_ids)))
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
        boards = env.support_boxes or [([0., 0., float(env.sim.board_top_y[None]) - .005], [.55, .4, .01])]
        for i, (center, size) in enumerate(boards):
            path = "/World/Board" if len(boards) == 1 else f"/World/Board/Support_{i+1}"
            board = UsdGeom.Cube.Define(stage, path)
            board.CreateSizeAttr(1.0)
            board.AddTranslateOp().Set(Gf.Vec3d(*center))
            board.AddScaleOp().Set(Gf.Vec3f(*size))
            board.CreateDisplayColorAttr([Gf.Vec3f(0.35, 0.22, 0.13)])

    def add_group(self, piece, indices):
        path = "/World/Fruit"
        parent = translation = None
        if self.masses is not None:
            path += "/Whole" if piece == 0 else f"/Piece_{piece:03d}"
            parent = UsdGeom.Xform.Define(self.stage, path)
            visibility = parent.CreateVisibilityAttr()
            visibility.Set(UsdGeom.Tokens.invisible if self.frame else UsdGeom.Tokens.inherited, 0)
            visibility.Set(UsdGeom.Tokens.inherited, self.frame)
            translation = parent.AddTranslateOp()
            parent.GetPrim().CreateAttribute("mpm:pieceId", Sdf.ValueTypeNames.Int).Set(piece)
            parent.GetPrim().CreateAttribute("mpm:massKg", Sdf.ValueTypeNames.Double).Set(
                float(self.masses[indices].sum(dtype=np.float64))
            )
        tissues = []
        for i, name in enumerate(self.env.record["material_order"], 1):
            subset = indices[self.env.tissue_ids[indices] == i]
            if not len(subset):
                continue
            node = UsdGeom.Points.Define(self.stage, path + "/" + name)
            node.CreateWidthsAttr([self.env.record["spacing_m"]])
            node.SetWidthsInterpolation("constant")
            node.CreateDisplayColorAttr([Gf.Vec3f(*self.palette[name])])
            if parent:
                node.CreateIdsAttr(Vt.Int64Array.FromNumpy(subset.astype(np.int64)))
            tissues.append((node.CreatePointsAttr(), subset))
        self.groups[piece] = (parent, translation, indices, tissues)

    def update(self, state):
        frame = self.frame
        self.time.Set(float(state["time_s"]), frame)
        piece_ids = state.get("piece_ids")
        active = [0] if piece_ids is None else np.unique(piece_ids).tolist()
        if piece_ids is not None:
            self.piece_count.Set(len(active), frame)
            for piece in active:
                if piece not in self.groups:
                    self.add_group(piece, np.flatnonzero(piece_ids == piece))
            if 0 not in active and 0 in self.groups:
                self.groups[0][0].CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible, frame)
        for piece in active:
            _, translation, indices, tissues = self.groups[piece]
            center = 0.
            if translation is not None:
                center = np.average(state["positions"][indices].astype(float), axis=0, weights=self.masses[indices])
                translation.Set(Gf.Vec3d(*center), frame)
            for attr, subset in tissues:
                attr.Set(points(state["positions"][subset] - center), frame)
        for path, matrix in state["rigid_rest_to_world"].items():
            self.rigid[path].Set(Gf.Matrix4d(*matrix.T.ravel().tolist()), frame)
        knife = state["knife"]
        self.translation.Set(Gf.Vec3d(*knife["position"].tolist()), frame)
        q = knife["quaternion"].tolist()
        self.orientation.Set(Gf.Quatf(q[0], Gf.Vec3f(*q[1:])), frame)
        self.stage.SetEndTimeCode(frame)
        self.frame += 1
