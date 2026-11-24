"""Attach upstream state to a Z-up, meter-scale Isaac USD stage after Kit starts."""
import numpy as np
from pxr import Gf, Sdf, UsdGeom, UsdShade, Vt
import trimesh

from .adapter import TO_WORLD
from .tissues import sample_interior_srgb, sample_pulp_srgb


def linear_color(srgb):
    return np.where(srgb <= .04045,srgb/12.92,((srgb+.055)/1.055)**2.4).astype(np.float32)


def surface_mesh(stage, path):
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr(True)
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    return mesh


def write_surface(mesh, points, faces, normals):
    mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points))
    mesh.GetNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(normals))
    mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray.FromNumpy(np.full(len(faces),3,np.int32)))
    mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(np.ascontiguousarray(faces.ravel())))
    extent = np.stack((points.min(0), points.max(0))) if len(points) else np.zeros((2,3),np.float32)
    mesh.GetExtentAttr().Set(Vt.Vec3fArray.FromNumpy(extent))


def tissue_primvars(mesh):
    api = UsdGeom.PrimvarsAPI(mesh)
    return (api.CreatePrimvar("tissueColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.uniform),
            api.CreatePrimvar("tissueId", Sdf.ValueTypeNames.IntArray, UsdGeom.Tokens.uniform),
            api.CreatePrimvar("segmentId", Sdf.ValueTypeNames.IntArray, UsdGeom.Tokens.uniform))


class SurfaceView:
    def __init__(self, stage, path, model):
        from .fruit_surface import FruitSurface
        self.surface = FruitSurface(model)
        self.pulp = None
        self.stem = None
        self.pulp_parts = {}
        if self.surface.layers:
            self.root = UsdGeom.Xform.Define(stage, path).GetPrim()
            self.mesh = surface_mesh(stage, path+"/Peel")
            pulp_path = path+"/Pulp"
            if self.surface.sectors:
                self.pulp = UsdGeom.Xform.Define(stage, pulp_path)
                for segment in range(1,self.surface.sectors["count"]+1):
                    self.pulp_parts[segment] = surface_mesh(stage, f"{pulp_path}/Segment_{segment:02d}")
            else:
                self.pulp = surface_mesh(stage, pulp_path)
                self.pulp_parts[0] = self.pulp
                for name in self.surface.interior_interfaces:
                    self.pulp_parts[name] = surface_mesh(stage,path+"/"+name)
            looks = UsdGeom.Scope.Define(stage, path+"/Looks").GetPath()
        else:
            self.mesh = surface_mesh(stage, path)
            self.root = self.mesh.GetPrim()
            looks = self.mesh.GetPath()
        self.st = UsdGeom.PrimvarsAPI(self.mesh).CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
        def material(name, color):
            mat = UsdShade.Material.Define(stage, looks.AppendChild(name))
            shader = UsdShade.Shader.Define(stage, mat.GetPath().AppendChild("Surface"))
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(.72)
            diffuse = shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)
            diffuse.Set(Gf.Vec3f(*color))
            mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            return mat, diffuse
        def use_tissue_color(mat, diffuse, primvar="tissueColor"):
            reader = UsdShade.Shader.Define(stage,mat.GetPath().AppendChild("TissueColor"))
            reader.CreateIdAttr("UsdPrimvarReader_float3")
            reader.CreateInput("varname",Sdf.ValueTypeNames.Token).Set(primvar)
            diffuse.ConnectToSource(reader.ConnectableAPI(),"result")
        if self.surface.stem_interface is not None:
            self.stem = surface_mesh(stage,path+"/Stem")
            color = self.surface.tissues["materials"][self.surface.tissues["stem_id"]-1]["srgb"]
            wood, _ = material("Stem",linear_color(np.asarray(color)/255.).tolist())
            UsdShade.MaterialBindingAPI.Apply(self.stem.GetPrim()).Bind(wood)
            self.stem_id = UsdGeom.PrimvarsAPI(self.stem).CreatePrimvar(
                "tissueId", Sdf.ValueTypeNames.IntArray, UsdGeom.Tokens.uniform)
        peel, diffuse = material("Peel", (.76,.50,.075))
        if self.surface.texture:
            reader = UsdShade.Shader.Define(stage, peel.GetPath().AppendChild("UV"))
            reader.CreateIdAttr("UsdPrimvarReader_float2")
            reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
            tex = UsdShade.Shader.Define(stage, peel.GetPath().AppendChild("Texture"))
            tex.CreateIdAttr("UsdUVTexture")
            tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(self.surface.texture)))
            tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
            for axis in ("wrapS","wrapT"):
                tex.CreateInput(axis,Sdf.ValueTypeNames.Token).Set("repeat")
            tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
            diffuse.ConnectToSource(tex.ConnectableAPI(), "rgb")
        binding = UsdShade.MaterialBindingAPI.Apply(self.mesh.GetPrim())
        binding.Bind(peel)
        self.cut = binding.CreateMaterialBindSubset("CutFaces", Vt.IntArray())
        if self.stem:
            self.scar = binding.CreateMaterialBindSubset("StemScar",Vt.IntArray())
            self.scar_mat,scar_color = material("StemScar",(.25,.12,.05))
            use_tissue_color(self.scar_mat,scar_color)
            self.scar_binding = UsdShade.MaterialBindingAPI.Apply(self.scar.GetPrim())
        self.flesh, flesh_color = material("Flesh", (.86,.73,.47))
        if self.surface.tissues is not None:
            self.color, self.tissue_id, self.segment_id = tissue_primvars(self.mesh)
            if self.pulp:
                UsdShade.MaterialBindingAPI.Apply(self.pulp.GetPrim()).Bind(self.flesh)
                self.pulp_primvars = {key: tissue_primvars(mesh) for key, mesh in self.pulp_parts.items()}
                self.has_pulp_color = "pulp_srgb" in self.surface.tissues
                if self.has_pulp_color:
                    for color, _, _ in self.pulp_primvars.values():
                        color.SetInterpolation(UsdGeom.Tokens.vertex)
                    shader = UsdShade.Shader.Get(stage,self.flesh.GetPath().AppendChild("Surface"))
                    pulp_material = self.surface.tissues["materials"][self.surface.layers["pulp"]-1]
                    shader.GetInput("roughness").Set(pulp_material.get("roughness", .44))
                    shader.CreateInput("ior",Sdf.ValueTypeNames.Float).Set(1.38)
                if self.surface.sectors:
                    # Bind directly to the animated wedge meshes. Their vertex colors
                    # use a separate shader from the uniform face colors of cut peel.
                    pulp_row = self.surface.tissues["materials"][self.surface.layers["pulp"]-1]
                    segments, segment_color = material("Segments",linear_color(np.array(pulp_row["srgb"])/255.).tolist())
                    segment_shader = UsdShade.Shader.Get(stage,segments.GetPath().AppendChild("Surface"))
                    segment_shader.GetInput("roughness").Set(pulp_row.get("roughness",.44))
                    segment_shader.CreateInput("ior",Sdf.ValueTypeNames.Float).Set(1.38)
                    use_tissue_color(segments,segment_color,"displayColor")
                    self.segment_colors = {}
                    interpolation = UsdGeom.Tokens.vertex if self.has_pulp_color else UsdGeom.Tokens.uniform
                    for key,mesh in self.pulp_parts.items():
                        self.segment_colors[key] = mesh.CreateDisplayColorPrimvar(interpolation)
                        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(segments)
            use_tissue_color(self.flesh,flesh_color)
            srgb = np.array([m["srgb"] for m in self.surface.tissues["materials"]])/255.
            self.palette = linear_color(srgb)
            for name,tissue in self.surface.tissues.get("interior_meshes",{}).items():
                row = self.surface.tissues["materials"][tissue-1]
                mat, diffuse = material(name,self.palette[tissue-1].tolist())
                use_tissue_color(mat,diffuse)
                shader = UsdShade.Shader.Get(stage,mat.GetPath().AppendChild("Surface"))
                shader.GetInput("roughness").Set(row.get("roughness",.54))
                UsdShade.MaterialBindingAPI.Apply(self.pulp_parts[name].GetPrim()).Bind(mat)

        self.peel = peel
        self.cut_binding = UsdShade.MaterialBindingAPI.Apply(self.cut.GetPrim())
        self.cut_binding.Bind(peel)
        self.has_interior = False

    @property
    def pulp_mesh(self):
        """Legacy layer handle. Segmented pulp exposes its meshes under the group."""
        return self.pulp

    def update(self, positions):
        if self.pulp:
            peel, fields = self.surface.layer_fields(positions)
            if self.stem:
                stem, _ = self.surface.triangulate(*self.surface.stem_fields,textured=False)
                write_surface(self.stem,*stem[:3])
                ids = np.full(len(stem[1]),self.surface.tissues["stem_id"],np.int32)
                self.stem_id.Set(Vt.IntArray.FromNumpy(ids))
            for key, (geometry, ids, segments, rest) in self.surface.pulp_parts(*fields,include_material=True):
                write_surface(self.pulp_parts[key], *geometry[:3])
                color, tissue_id, segment_id = self.pulp_primvars[key]
                if key in self.surface.interior_interfaces:
                    tissue = self.surface.tissues["interior_meshes"][key]
                    rgb = linear_color(sample_interior_srgb(self.surface.tissues,rest,tissue))
                elif self.has_pulp_color:
                    rgb = linear_color(sample_pulp_srgb(self.surface.tissues,rest))
                else:
                    rgb = self.palette[ids-1]
                values = Vt.Vec3fArray.FromNumpy(rgb)
                color.Set(values)
                if self.surface.sectors:
                    self.segment_colors[key].Set(values)
                tissue_id.Set(Vt.IntArray.FromNumpy(ids.astype(np.int32)))
                segment_id.Set(Vt.IntArray.FromNumpy(segments.astype(np.int32)))
            points, faces, normals, uv, interior = peel
        else:
            points, faces, normals, uv, interior = self.surface.reconstruct(positions)
        write_surface(self.mesh, points, faces, normals)
        self.st.Set(Vt.Vec2fArray.FromNumpy(uv))
        if self.surface.tissues is not None:
            tissue = np.ones(len(faces),np.int32)
            segment = np.zeros(len(faces),np.int32)
            if self.pulp:
                tissue.fill(self.surface.layers["peel"])
            else:
                tissue[interior] = self.surface.cut_tissue_ids
                segment[interior] = self.surface.cut_segment_ids
            colors = self.palette[tissue-1]
            if self.stem:
                scar = self.surface.scar_faces
                # Removing the stalk reveals a small cork-colored attachment on the peel.
                rgb = np.array([153,100,64]) - self.surface.scar_radius[:,None]*[32,28,20]
                colors[scar] = linear_color(rgb/255.)
                self.scar.GetIndicesAttr().Set(Vt.IntArray.FromNumpy(scar))
                self.scar_binding.Bind(self.scar_mat if len(scar) else self.peel)
                interior = np.setdiff1d(interior,scar,assume_unique=True)
            self.color.Set(Vt.Vec3fArray.FromNumpy(colors))
            self.tissue_id.Set(Vt.IntArray.FromNumpy(tissue))
            self.segment_id.Set(Vt.IntArray.FromNumpy(segment))
        self.cut.GetIndicesAttr().Set(Vt.IntArray.FromNumpy(interior))
        # RTX can apply a bound material to the whole mesh when a subset is empty.
        has_interior = bool(interior.size)
        if has_interior != self.has_interior:
            self.cut_binding.Bind(self.flesh if has_interior else self.peel)
            self.has_interior = has_interior


class IsaacCuttingView:
    def __init__(self, stage, scene, root="/World/CulinaryCut", *, fruit_display=None):
        if fruit_display is None:
            fruit_display = "surface" if "fruit" in scene else "particles"
        if fruit_display not in ("surface", "particles"):
            raise ValueError("fruit_display must be 'surface' or 'particles'")
        if UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.z or not np.isclose(UsdGeom.GetStageMetersPerUnit(stage), 1.):
            raise ValueError("CulinaryCut requires a Z-up stage with metersPerUnit=1")
        if stage.GetPrimAtPath(root).IsValid():
            raise ValueError(f"USD prim already exists at {root}")
        UsdGeom.Xform.Define(stage, root)
        if "rigid_pit" in scene:
            pit = scene["rigid_pit"]
            core = surface_mesh(stage, root+"/RigidPit")
            solid = trimesh.Trimesh(pit["vertices"], pit["faces"], process=False)
            write_surface(core, pit["vertices"].astype(np.float32), pit["faces"].astype(np.int32),
                          solid.vertex_normals.astype(np.float32))
            core.CreateDisplayColorAttr([Gf.Vec3f(.28,.095,.035)])

        def mesh(path, geometry, color, *, static=False):
            source = trimesh.load(geometry["mesh"], force="mesh")
            vertices = trimesh.transform_points(source.vertices, geometry["asset_to_sdf"])
            if static:
                vertices[:, 1] += geometry["offset_y"]
                vertices = vertices @ TO_WORLD.T
            obj = UsdGeom.Mesh.Define(stage, path)
            obj.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(vertices, dtype=np.float32)))
            obj.CreateFaceVertexCountsAttr([3]*len(source.faces))
            obj.CreateFaceVertexIndicesAttr(source.faces.ravel().tolist())
            obj.CreateDoubleSidedAttr(True)
            obj.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
            obj.CreateDisplayColorAttr([Gf.Vec3f(*color)])

        knife = UsdGeom.Xform.Define(stage, root+"/Knife")
        self.translation = knife.AddTranslateOp()
        self.orientation = knife.AddOrientOp()
        mesh(root+"/Knife/Mesh", scene["knife"], (.50,.55,.59))
        if scene["board"]:
            mesh(root+"/Board", scene["board"], (.45,.27,.12), static=True)
        self.surface_view = None
        if fruit_display == "surface":
            self.surface_view = SurfaceView(stage, root+"/Fruit", scene["fruit"])
            self.positions = self.surface_view.root.CreateAttribute(
                "mpm:positions", Sdf.ValueTypeNames.Point3fArray)
            return
        fruit = UsdGeom.PointInstancer.Define(stage, root+"/Fruit")
        prototype = UsdGeom.Sphere.Define(stage, root+"/Fruit/Particle")
        prototype.CreateRadiusAttr(scene["particle_radius"])
        prototype.CreateDisplayColorAttr([Gf.Vec3f(.95,.77,.31)])
        fruit.CreatePrototypesRel().SetTargets([prototype.GetPath()])
        self.positions = fruit.CreatePositionsAttr()
        self.indices = fruit.CreateProtoIndicesAttr()
        self.count = None

    def update(self, fruit, knife):
        if fruit["time"] != knife["time"]:
            raise ValueError("Fruit and knife timestamps differ")
        if self.surface_view is not None:
            self.surface_view.update(fruit["positions"])
        elif fruit["count"] != self.count:
            self.count = fruit["count"]
            self.indices.Set(Vt.IntArray.FromNumpy(np.zeros(self.count, np.int32)))
        self.positions.Set(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(fruit["positions"], dtype=np.float32)))
        self.translation.Set(Gf.Vec3d(*knife["position"]))
        q = knife["quaternion"]
        self.orientation.Set(Gf.Quatf(float(q[0]), Gf.Vec3f(*q[1:])))
