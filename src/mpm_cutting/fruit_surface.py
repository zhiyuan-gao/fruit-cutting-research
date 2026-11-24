"""Reconstruct the advected asset surface and exposed flesh from MPM material points."""
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp
try:
    from warp.geometry import IsoSurfaceMarchingCubes
except ImportError:  # Isaac Sim 5.1 ships Warp 1.8 in Kit.
    IsoSurfaceMarchingCubes = None

from .adapter import TO_WORLD


def asset_triangles(model):
    """Keep OBJ face-corner UVs, including seams on the original fruit texture."""
    path = Path(model["mesh"])
    vertices, coords, triangles, corner_uv = [], [], [], []
    texture = None
    for line in path.read_text().splitlines():
        words = line.split()
        if not words:
            continue
        if words[0] == "v":
            vertices.append([float(v) for v in words[1:4]])
        elif words[0] == "vt":
            coords.append([float(v) for v in words[1:3]])
        elif words[0] == "mtllib":
            mtl = path.parent / " ".join(words[1:])
            if mtl.is_file():
                for row in mtl.read_text().splitlines():
                    key, _, name = row.strip().partition(" ")
                    if key == "map_Kd":
                        texture = mtl.parent / name.strip()
        elif words[0] == "f":
            corners = [w.split("/") for w in words[1:]]
            for j in range(1, len(corners)-1):
                tri = [corners[k] for k in (0, j, j+1)]
                def index(s, size):
                    n = int(s)
                    return n-1 if n > 0 else size+n
                triangles.append([index(c[0], len(vertices)) for c in tri])
                corner_uv.extend(coords[index(c[1], len(coords))] if len(c)>1 and c[1] else (0.,0.) for c in tri)
    transform = model["asset_to_sdf"]
    verts = (np.asarray(vertices) @ transform[:3,:3].T + transform[:3,3]) @ TO_WORLD.T
    verts[:,2] += model["lift"]
    return verts, triangles, corner_uv, texture


@wp.func
def sample(grid: wp.array3d(dtype=Any), q: wp.vec3):
    # Scalar or vector trilinear interpolation, in grid-index coordinates.
    x = wp.clamp(q[0], 0., float(grid.shape[0]-1))
    y = wp.clamp(q[1], 0., float(grid.shape[1]-1))
    z = wp.clamp(q[2], 0., float(grid.shape[2]-1))
    i = wp.min(int(x), grid.shape[0]-2)
    j = wp.min(int(y), grid.shape[1]-2)
    k = wp.min(int(z), grid.shape[2]-2)
    a, b, c = x-float(i), y-float(j), z-float(k)
    return wp.lerp(wp.lerp(wp.lerp(grid[i,j,k], grid[i+1,j,k], a),
                           wp.lerp(grid[i,j+1,k], grid[i+1,j+1,k], a), b),
                   wp.lerp(wp.lerp(grid[i,j,k+1], grid[i+1,j,k+1], a),
                           wp.lerp(grid[i,j+1,k+1], grid[i+1,j+1,k+1], a), b), c)


@wp.func
def material_at(grid: wp.uint64, p: wp.vec3, spacing: float, rest: wp.array(dtype=wp.vec3),
                current: wp.array(dtype=wp.vec3)):
    density = float(0.)
    offset = wp.vec3(0.)
    radius = 2.4*spacing
    for idx in wp.hash_grid_query(grid, p, radius):
        d = current[idx]-p
        r2 = wp.dot(d,d)/(spacing*spacing)
        if r2 < 2.4*2.4:
            w = wp.exp(-r2/(2.*.62*.62))
            density += w
            offset += w*(current[idx]-rest[idx])
    return density, offset/wp.max(density, 1.e-12)


@wp.kernel
def reference_density(grid: wp.uint64, points: wp.array(dtype=wp.vec3), origin: wp.vec3,
                      step: float, spacing: float, density: wp.array3d(dtype=float)):
    i,j,k = wp.tid()
    p = origin + step*wp.vec3(float(i),float(j),float(k))
    rho, _ = material_at(grid,p,spacing,points,points)
    density[i,j,k] = rho


@wp.kernel
def evaluate(grid: wp.uint64, rest: wp.array(dtype=wp.vec3), current: wp.array(dtype=wp.vec3),
             origin: wp.vec3, step: float, spacing: float, bulk_shift: wp.vec3,
             sdf: wp.array3d(dtype=float), asset_origin: wp.vec3, voxel: float,
             density: wp.array3d(dtype=float), density_origin: wp.vec3,
             field: wp.array3d(dtype=float), offsets: wp.array3d(dtype=wp.vec3)):
    i,j,k = wp.tid()
    p = origin + step*wp.vec3(float(i),float(j),float(k))
    rho, displacement = material_at(grid,p,spacing,rest,current)
    if rho < 1.e-8:
        displacement = bulk_shift
    material = p-displacement
    skin = sample(sdf, (material-asset_origin)/voxel)
    baseline = sample(density,(material-density_origin)/step)
    value = skin
    if baseline > .08:
        # A void relative to the same material at rest exposes an internal face.
        empty = (.52 - rho/baseline)*spacing
        value = wp.max(skin,empty)
    field[i,j,k] = value
    offsets[i,j,k] = displacement


@wp.kernel
def split_layers(whole: wp.array3d(dtype=float), offsets: wp.array3d(dtype=wp.vec3),
                 origin: wp.vec3, step: float, rest_interface: wp.array3d(dtype=float),
                 to_grid: wp.mat33, translation: wp.vec3,
                 peel: wp.array3d(dtype=float), pulp: wp.array3d(dtype=float)):
    i,j,k = wp.tid()
    rest = origin + step*wp.vec3(float(i),float(j),float(k)) - offsets[i,j,k]
    interface = sample(rest_interface, to_grid*rest + translation)
    peel[i,j,k] = wp.max(whole[i,j,k], -interface)
    pulp[i,j,k] = wp.max(whole[i,j,k], interface)


@wp.kernel
def segment_field(pulp: wp.array3d(dtype=float), offsets: wp.array3d(dtype=wp.vec3),
                  origin: wp.vec3, step: float, to_local: wp.mat33, translation: wp.vec3,
                  start: float, sweep: float, twist: float, radius: float,
                  rest_interface: wp.array3d(dtype=float), to_grid: wp.mat33, grid_shift: wp.vec3,
                  field: wp.array3d(dtype=float)):
    i,j,k = wp.tid()
    rest = origin + step*wp.vec3(float(i),float(j),float(k)) - offsets[i,j,k]
    local = to_local*rest + translation
    a = start - twist*local[1]
    b = a + sweep
    side = radius*wp.max(local[0]*wp.sin(a)-local[2]*wp.cos(a),
                         local[2]*wp.cos(b)-local[0]*wp.sin(b))
    # Narrow seams and rounded shoulders keep neighboring membranes distinguishable.
    side += .005*radius
    outer = sample(rest_interface, to_grid*rest + grid_shift)
    rounding = .05*radius
    t = wp.max(rounding-wp.abs(outer-side), 0.)
    shoulder = wp.max(outer,side) + t*t/(4.*rounding)
    field[i,j,k] = wp.max(pulp[i,j,k], shoulder)


@wp.kernel
def grid_to_world(vertices: wp.array(dtype=wp.vec3), origin: wp.vec3, step: float):
    i = wp.tid()
    vertices[i] = origin+step*vertices[i]


@wp.kernel
def vertex_data(vertices: wp.array(dtype=wp.vec3), field: wp.array3d(dtype=float),
                offsets: wp.array3d(dtype=wp.vec3), origin: wp.vec3, step: float,
                material: wp.array(dtype=wp.vec3), normals: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    q = (vertices[i]-origin)/step
    material[i] = vertices[i]-sample(offsets,q)
    grad = wp.vec3(0.)
    for axis in range(3):
        delta = wp.vec3(0.)
        delta[axis] = .75
        grad[axis] = sample(field,q+delta)-sample(field,q-delta)
    normals[i] = wp.normalize(grad)


@wp.kernel
def face_data(asset: wp.uint64, asset_uv: wp.array(dtype=wp.vec2),
              material: wp.array(dtype=wp.vec3), faces: wp.array(dtype=int),
              sdf: wp.array3d(dtype=float), asset_origin: wp.vec3, voxel: float, step: float,
              uv: wp.array(dtype=wp.vec2), flesh: wp.array(dtype=int)):
    i = wp.tid()
    a,b,c = faces[3*i],faces[3*i+1],faces[3*i+2]
    center = (material[a]+material[b]+material[c])/3.
    if sample(sdf,(center-asset_origin)/voxel) < -.65*step:
        flesh[i] = 1
    else:
        hit = wp.mesh_query_point_no_sign(asset,center,100.)
        if hit.result:
            base = hit.face*3
            p0 = wp.mesh_get_point(asset,base)
            e1 = wp.mesh_get_point(asset,base+1)-p0
            e2 = wp.mesh_get_point(asset,base+2)-p0
            aa,ab,bb = wp.dot(e1,e1),wp.dot(e1,e2),wp.dot(e2,e2)
            det = wp.max(aa*bb-ab*ab,1.e-22)
            for j in range(3):
                d = material[faces[3*i+j]]-p0
                s = wp.clamp((bb*wp.dot(e1,d)-ab*wp.dot(e2,d))/det,0.,1.)
                t = wp.clamp((aa*wp.dot(e2,d)-ab*wp.dot(e1,d))/det,0.,1.-s)
                uv[3*i+j] = (1.-s-t)*asset_uv[base]+s*asset_uv[base+1]+t*asset_uv[base+2]


def smooth_shading_normals(normals, triangles, radius_in_cells):
    """Diffuse lighting on connected faces. Positions and silhouette stay on the measured surface."""
    from scipy.sparse import csr_matrix

    i = triangles.ravel()
    j = triangles[:,[1,2,0]].ravel()
    alignment = np.einsum("ij,ij->i",normals[i],normals[j])
    # Break the smoothing stencil at folds and where the two sides of a cut meet.
    weights = np.clip((alignment-.64)/.32,0.,1.).astype(np.float32)
    edges = csr_matrix((weights,(i,j)),shape=(len(normals),len(normals)))
    edges = edges + edges.T
    total = np.asarray(edges.sum(axis=1)).ravel()
    edges = edges.multiply((1./np.maximum(total,1.e-8))[:,None]).tocsr()
    result = normals.copy()
    for _ in range(min(48,max(1,round(2*radius_in_cells**2)))):
        result = .15*result + .85*(edges @ result)
    result = .92*result + .08*normals
    length = np.linalg.norm(result,axis=1,keepdims=True)
    return np.where(length>1.e-6,result/np.maximum(length,1.e-6),normals)


class FruitSurface:
    def __init__(self, model):
        self.device = "cuda"
        self.spacing = float(model["spacing"])
        self.step = self.spacing / 1.5
        self.voxel = float(model["voxel"])
        self.tissues = model.get("tissues")
        self.layers = self.tissues.get("surface_layers") if self.tissues else None
        self.interior_interfaces = {}
        self.stem_interface = None
        self.shading_radius = 0.
        skin_sdf = model["sdf"]
        soft_sdf = model.get("soft_sdf")
        if self.layers:
            pitch = float(self.tissues["voxel"])
            tf = self.tissues["world_to_asset"]
            world_pitch = pitch / np.linalg.norm(tf[:3,:3], ord=2)
            if "peel_interface_vox" in self.tissues:
                distance = self.tissues["peel_interface_vox"].astype(np.float32)
            else:
                from scipy.ndimage import distance_transform_edt
                labels = self.tissues["labels"]
                inside = (labels != 0) & (labels != self.layers["peel"])
                distance = distance_transform_edt(~inside).astype(np.float32)
                distance -= distance_transform_edt(inside).astype(np.float32)
            interface = distance*world_pitch
            self.shading_radius = self.tissues.get("display_normal_smoothing_m",0.) * world_pitch/pitch
            if "stem_id" in self.tissues:
                self.stem_interface = self.array(self.tissues["stem_interface_vox"].astype(np.float32)*world_pitch,wp.float32)
            self.interface = self.array(interface, wp.float32)
            if self.tissues.get("interior_meshes"):
                from scipy.ndimage import distance_transform_edt, gaussian_filter
                for name,tissue in self.tissues["interior_meshes"].items():
                    inside = self.tissues["labels"] >= tissue
                    if "stem_id" in self.tissues:
                        inside &= self.tissues["labels"] != self.tissues["stem_id"]
                    signed = distance_transform_edt(~inside) - distance_transform_edt(inside)
                    signed = gaussian_filter(signed,.6)
                    self.interior_interfaces[name] = self.array(signed*world_pitch,wp.float32)
            self.to_label_grid = wp.mat33(*(tf[:3,:3]/pitch).ravel())
            self.label_translation = wp.vec3(*((tf[:3,3]-self.tissues["origin"])/pitch))
            self.step = min(self.step, 1.5*world_pitch)
        self.initial = model["rest"]
        self.initial_center = self.initial.mean(axis=0, dtype=np.float64)
        self.rest = self.array(self.initial, wp.vec3)
        self.current = self.array(self.initial, wp.vec3)
        points, indices, st, self.texture = asset_triangles(model)
        self.sectors = self.tissues.get("radial_segments") if self.tissues else None
        if self.sectors:
            # Match the OBJ-bound normalization used to label the original MPM material.
            asset_points = points @ tf[:3,:3].T + tf[:3,3]
            half = np.ptp(asset_points, axis=0)/2
            center = asset_points.min(0) + half
            to_local = tf[:3,:3] / half[:,None]
            self.to_sector = wp.mat33(*to_local.ravel())
            self.sector_translation = wp.vec3(*((tf[:3,3]-center)/half))
            self.sector_radius = 1./np.linalg.norm(to_local[[0,2]], ord=2)
        self.asset = wp.Mesh(points=self.array(points, wp.vec3),indices=self.array(np.ravel(indices),wp.int32))
        self.uv = self.array(st,wp.vec2)
        orig = model["origin"]
        self.asset_origin = wp.vec3(float(orig[0]), -float(orig[2]+(model["sdf"].shape[0]-1)*self.voxel),
                                    float(orig[1]+model["lift"]))
        self.sdf = self.array(skin_sdf[::-1].transpose(2,0,1),wp.float32)
        self.soft_sdf = (self.array(soft_sdf[::-1].transpose(2,0,1),wp.float32)
                         if soft_sdf is not None else self.sdf)
        self.origin = np.floor((np.minimum(points.min(0), self.initial.min(0))-4*self.spacing)/self.step)*self.step
        upper = np.maximum(points.max(0),self.initial.max(0))+4*self.spacing
        self.shape = tuple((np.ceil((upper-self.origin)/self.step).astype(int)+1).tolist())
        self.grid = wp.HashGrid(64,96,48,device=self.device)
        self.cubes = None
        self.capacity = max(8192,len(self.initial)*2)
        self.grid.build(self.rest,2.4*self.spacing)
        self.density = wp.empty(self.shape,dtype=wp.float32,device=self.device)
        self.launch(reference_density,self.shape,[self.grid.id,self.rest,wp.vec3(*self.origin),self.step,
                                                 self.spacing,self.density])

    def array(self, values, dtype):
        return wp.array(np.ascontiguousarray(values),dtype=dtype,device=self.device)

    def launch(self, kernel, size, args):
        wp.launch(kernel,dim=size,inputs=args,device=self.device)

    def extract(self, field, lo):
        shape = field.shape
        if IsoSurfaceMarchingCubes is not None:
            upper = lo+(np.array(shape)-1)*self.step
            return IsoSurfaceMarchingCubes.extract(field,lower=tuple(lo),upper=tuple(upper))
        while True:
            if self.cubes is None:
                self.cubes = wp.MarchingCubes(*shape,self.capacity,2*self.capacity,device=self.device)
            self.cubes.resize(*shape,self.capacity,2*self.capacity)
            try:
                self.cubes.surface(field,0.)
                break
            except RuntimeError as exc:
                if "Buffers may not be large enough" not in str(exc):
                    raise
                self.capacity *= 2
                self.cubes = None
        # The 1.8 CUDA extractor puts sample (0,0,0) at the grid-cell center.
        self.launch(grid_to_world,len(self.cubes.verts),[self.cubes.verts,wp.vec3(*(lo-.5*self.step)),self.step])
        return self.cubes.verts, self.cubes.indices

    def evaluate_positions(self, positions):
        if positions.shape != self.initial.shape or not np.isfinite(positions).all():
            raise ValueError("Surface reconstruction requires finite positions in the original particle order")
        self.current.assign(positions)
        self.grid.build(self.current,2.4*self.spacing)
        shift = positions.mean(axis=0,dtype=np.float64)-self.initial_center
        # Grow the domain for separated pieces; follow a translation without cropping the tips.
        lo = np.minimum(self.origin+shift, positions.min(0)-3*self.spacing)
        hi = np.maximum(self.origin+shift+(np.array(self.shape)-1)*self.step, positions.max(0)+3*self.spacing)
        shape = tuple((np.ceil((hi-lo)/self.step-1.e-6).astype(int)+1).tolist())
        origin = wp.vec3(*lo)
        field = wp.empty(shape,dtype=wp.float32,device=self.device)
        offsets = wp.empty(shape,dtype=wp.vec3,device=self.device)
        self.launch(evaluate,shape,[self.grid.id,self.rest,self.current,origin,self.step,self.spacing,
                    wp.vec3(*shift),self.soft_sdf,self.asset_origin,self.voxel,self.density,wp.vec3(*self.origin),
                    field,offsets])
        return field, offsets, lo

    def triangulate(self, field, offsets, lo, *, textured, smooth_normals=False):
        origin = wp.vec3(*lo)
        vertices,faces = self.extract(field,lo)
        n = len(vertices)
        if not n or not len(faces):
            empty = np.empty((0, 3), np.float32)
            geometry = (empty, np.empty((0, 3), np.int32), empty,
                        np.empty((0, 2), np.float32), np.empty(0, np.int32))
            return geometry, empty
        material = wp.empty(n,dtype=wp.vec3,device=self.device)
        normals = wp.empty_like(material)
        self.launch(vertex_data,n,[vertices,field,offsets,origin,self.step,material,normals])
        points, tris, gradients = vertices.numpy(), faces.numpy().reshape(-1,3), normals.numpy()
        if textured:
            uv = wp.zeros(len(faces),dtype=wp.vec2,device=self.device)
            flesh = wp.zeros(len(faces)//3,dtype=wp.int32,device=self.device)
            self.launch(face_data,len(flesh),[self.asset.id,self.uv,material,faces,self.sdf,
                                             self.asset_origin,self.voxel,self.step,uv,flesh])
            st = uv.numpy().reshape(-1,3,2)
            interior = np.flatnonzero(flesh.numpy()).astype(np.int32)
        else:
            st = np.empty((0,3,2),np.float32)
            interior = np.empty(0,np.int32)
        # Keep the extractor's connected topology. Orient the whole mesh consistently.
        probe = tris[::64]
        a,b,c = points[probe].transpose(1,0,2)
        if np.sum(np.cross(b-a,c-a) * gradients[probe].sum(axis=1)) < 0:
            tris = tris[:, [0,2,1]]
            if textured:
                st = st[:, [0,2,1]]
        if smooth_normals and self.shading_radius:
            gradients = smooth_shading_normals(gradients,tris,self.shading_radius/self.step)
        return (points, tris, gradients, st.reshape(-1,2), interior), material.numpy() if self.tissues else None

    def reconstruct(self, positions):
        geometry, material = self.triangulate(*self.evaluate_positions(positions), textured=True)
        points, tris, _, _, interior = geometry
        if not len(points):
            raise ValueError("The fruit has no reconstructed surface")
        if self.tissues is not None:
            from .tissues import sample_tissues
            centers = material[tris[interior]].mean(axis=1)
            self.cut_tissue_ids, self.cut_segment_ids = sample_tissues(self.tissues, centers, soft_only=True)
        return geometry

    def layer_fields(self, positions):
        whole, offsets, lo = self.evaluate_positions(positions)
        self.whole_fields = whole, offsets, lo
        if self.stem_interface is not None:
            body, stem = wp.empty_like(whole), wp.empty_like(whole)
            self.launch(split_layers,whole.shape,[whole,offsets,wp.vec3(*lo),self.step,self.stem_interface,
                                                 self.to_label_grid,self.label_translation,body,stem])
            self.stem_fields = stem, offsets, lo
            whole = body
        peel, pulp = wp.empty_like(whole), wp.empty_like(whole)
        self.launch(split_layers,whole.shape,[whole,offsets,wp.vec3(*lo),self.step,self.interface,
                                             self.to_label_grid,self.label_translation,peel,pulp])
        peel_mesh, material = self.triangulate(peel,offsets,lo,textured=True,smooth_normals=True)
        if self.stem_interface is not None:
            socket = self.tissues["stem_socket"]
            tf = self.tissues["world_to_asset"]
            centers = material[peel_mesh[1]].mean(axis=1) @ tf[:3,:3].T + tf[:3,3]
            relative = centers-np.asarray(socket["base_center_m"])
            radius = np.linalg.norm(relative,axis=1)/socket["scar_radius_m"]
            self.scar_faces = np.flatnonzero((radius<=1.) & (relative[:,1]>-.0013)).astype(np.int32)
            self.scar_radius = radius[self.scar_faces]
        return peel_mesh, (pulp, offsets, lo)

    def pulp_geometry(self, field, offsets, lo, *, segment_id=None, include_material=False, smooth_normals=True):
        from .tissues import sample_tissues
        pulp_mesh, material = self.triangulate(field,offsets,lo,textured=False,smooth_normals=smooth_normals)
        centers = material[pulp_mesh[1]].mean(axis=1)
        tissue, segment = sample_tissues(self.tissues,centers,soft_only=True)
        # Interface centers can round to the peel voxel on the other side.
        tissue[tissue == self.layers["peel"]] = self.layers["pulp"]
        if segment_id is not None:
            segment = np.full(len(tissue),segment_id,np.int32)
        result = pulp_mesh, tissue, segment
        return (*result, material) if include_material else result

    def reconstruct_layers(self, positions):
        peel, fields = self.layer_fields(positions)
        return peel, *self.pulp_geometry(*fields)

    def pulp_parts(self, pulp, offsets, lo, *, include_material=False):
        if self.interior_interfaces:
            parent = pulp
            key = 0
            tissue = self.layers["pulp"]
            for name,interface in self.interior_interfaces.items():
                outside,inside = wp.empty_like(pulp),wp.empty_like(pulp)
                self.launch(split_layers,pulp.shape,[parent,offsets,wp.vec3(*lo),self.step,interface,
                            self.to_label_grid,self.label_translation,outside,inside])
                result = self.pulp_geometry(outside,offsets,lo,include_material=include_material,smooth_normals=key==0)
                result[1].fill(tissue)
                yield key,result
                parent,key,tissue = inside,name,self.tissues["interior_meshes"][name]
            result = self.pulp_geometry(parent,offsets,lo,include_material=include_material,smooth_normals=False)
            result[1].fill(tissue)
            yield key,result
            return
        if not self.sectors:
            yield 0, self.pulp_geometry(pulp,offsets,lo,include_material=include_material)
            return
        count = self.sectors["count"]
        sweep = 2*np.pi/count
        field = wp.empty_like(pulp)
        for segment in range(1,count+1):
            start = (segment-1)*sweep-self.sectors["phase_rad"]
            self.launch(segment_field,pulp.shape,[pulp,offsets,wp.vec3(*lo),self.step,self.to_sector,
                        self.sector_translation,start,sweep,self.sectors["twist_rad"],self.sector_radius,
                        self.interface,self.to_label_grid,self.label_translation,field])
            yield segment, self.pulp_geometry(field,offsets,lo,segment_id=segment,include_material=include_material)
