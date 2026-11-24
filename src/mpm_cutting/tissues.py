"""Sample asset-local tissue labels at material coordinates and initialize native MPM fields."""
import hashlib
import json
from pathlib import Path

import numpy as np


def load_tissues(description):
    path = Path(description)
    spec = json.loads(path.read_text())
    materials = spec["materials"]
    with np.load(path.parent/spec["volume"], allow_pickle=False) as data:
        volume = {key: data[key] for key in ("labels","lobes","origin","voxel","mesh_sha256")}
        if "peel_interface_vox" in data:
            volume["peel_interface_vox"] = data["peel_interface_vox"]
            if volume["peel_interface_vox"].shape != volume["labels"].shape:
                raise ValueError("Peel interface and tissue labels have different grids")
        if "stem" in spec:
            field = data["stem_interface_vox"]
            socket = spec["stem"]
            center = np.asarray(socket["base_center_m"],float)
            radius = float(socket["scar_radius_m"])
            if (field.shape != volume["labels"].shape or materials[-1]["name"] != "Stem"
                    or not np.isfinite(field).all() or not field.min() < 0 < field.max()
                    or center.shape!=(3,) or not np.isfinite(center).all() or not np.isfinite(radius) or radius<=0):
                raise ValueError("Invalid stem partition")
            volume["stem_interface_vox"] = field
            volume["stem_id"] = len(materials)
            volume["stem_socket"] = socket
        if "pulp_srgb" in data:
            field = data["pulp_srgb"]
            if field.shape != volume["labels"].shape+(3,) or field.dtype != np.uint8:
                raise ValueError("Expected an RGB8 color at each tissue voxel")
            volume["pulp_srgb"] = field
        if "interior_grain" in data:
            grain = data["interior_grain"]
            if grain.shape != volume["labels"].shape or grain.dtype != np.uint8:
                raise ValueError("Expected an 8-bit detail value per tissue voxel")
            volume["interior_grain"] = grain
        rigid_ids = [i for i,m in enumerate(materials,1) if m.get("rigid")]
        if rigid_ids:
            if len(rigid_ids) != 1:
                raise ValueError("Expected one independent pit")
            volume["rigid_id"], = rigid_ids
            volume["pit_vertices"] = data["pit_vertices"]
            volume["pit_faces"] = data["pit_faces"]
    labels, lobes = volume["labels"], volume["lobes"]
    if (labels.ndim != 3 or labels.shape != lobes.shape or labels.dtype != np.uint8 or lobes.dtype != np.uint8
            or volume["origin"].shape != (3,) or not np.isfinite(volume["origin"]).all()
            or not np.isfinite(volume["voxel"]) or volume["voxel"] <= 0
            or set(np.unique(labels)) != set(range(len(materials)+1))):
        raise ValueError("Invalid tissue volume or mismatched material table")
    rho, young, poisson = np.array([(m["density"],m["youngs_modulus"],m["poisson_ratio"])
                                   for m in materials if not m.get("rigid")]).T
    if (not np.isfinite([rho,young,poisson]).all() or min(rho) <= 0 or min(young) <= 0
            or min(poisson) <= -1 or max(poisson) >= .5):
        raise ValueError("Invalid elastic material parameters")
    volume["materials"] = materials
    if "mesh_variant" in spec:
        variant = spec["mesh_variant"]
        volume["mesh_variant"] = (str((path.parent/variant["mesh"]).resolve()), variant["source_sha256"])
    if "surface_layers" in spec:
        ids = {m["name"]: i for i, m in enumerate(materials, 1) if not m.get("rigid")}
        volume["surface_layers"] = {part: ids[spec["surface_layers"][part]] for part in ("peel", "pulp")}
        if volume["surface_layers"]["peel"] == volume["surface_layers"]["pulp"]:
            raise ValueError("Peel and pulp must have distinct tissue labels")
    smoothing = spec.get("display_normal_smoothing_m",spec.get("display_surface_smoothing_m"))
    if smoothing is not None:
        radius = float(smoothing)
        if "surface_layers" not in volume or not np.isfinite(radius) or radius <= 0:
            raise ValueError("Surface smoothing requires peel and pulp layers and a positive length")
        volume["display_normal_smoothing_m"] = radius
    if "interior_meshes" in spec:
        names = spec["interior_meshes"]
        nested_materials = materials[:-1] if "stem_id" in volume else materials
        ordered = [m["name"] for m in nested_materials]
        if (not names or not {"surface_layers","pulp_srgb","interior_grain"} <= volume.keys() or "radial_segments" in spec
                or ordered != [spec["surface_layers"]["peel"],spec["surface_layers"]["pulp"],*names]
                or len(set(ordered)) != len(ordered) or rigid_ids):
            raise ValueError("Nested soft meshes must follow the material table from surface to center")
        volume["interior_meshes"] = {name:ordered.index(name)+1 for name in names}
    if "radial_segments" in spec:
        sectors = spec["radial_segments"]
        if (not isinstance(sectors["count"], int) or sectors["count"] < 3
                or not np.array_equal(np.unique(lobes), np.arange(sectors["count"]+1))
                or not np.isfinite([sectors["phase_rad"], sectors["twist_rad"]]).all()
                or "surface_layers" not in volume):
            raise ValueError("Radial geometry and tissue segment labels must agree")
        volume["radial_segments"] = sectors
    return volume


def configure_tissue_timestep(cfg, volume):
    mesh = Path(cfg["cutting_mesh"]["mesh_path"])
    if hashlib.sha256(mesh.read_bytes()).hexdigest() != str(volume["mesh_sha256"]):
        raise ValueError(f"Tissue volume does not belong to {mesh}")
    world = cfg["world"]
    dx = (np.asarray(world["bounds_max"])-world["bounds_min"]).max()/world["grid_resolution"]
    for material in volume["materials"]:
        if material.get("rigid"):
            continue
        e, nu = material["youngs_modulus"], material["poisson_ratio"]
        wave = np.sqrt(e*(1-nu)/((1+nu)*(1-2*nu)*material["density"]))
        world["dt"] = min(world["dt"], .2*dx/wave)


def material_coordinates(asset_to_sdf, lift, to_world):
    """Homogeneous map from undeformed Isaac world coordinates to original OBJ coordinates."""
    sdf_to_world = np.eye(4)
    sdf_to_world[:3,:3] = to_world
    sdf_to_world[2,3] = lift
    return np.linalg.inv(sdf_to_world @ asset_to_sdf)


def voxel_coordinates(volume, rest_world):
    transform = volume["world_to_asset"]
    points = rest_world @ transform[:3,:3].T + transform[:3,3]
    return (points-volume["origin"])/volume["voxel"]


def sample_pulp_srgb(volume, rest_world):
    from scipy.ndimage import map_coordinates
    coords = voxel_coordinates(volume, rest_world).T
    field = volume["pulp_srgb"]
    return np.stack([map_coordinates(field[...,i],coords,order=1,mode="nearest",prefilter=False,
                                     output=np.float32) for i in range(3)],axis=1)/255.


def sample_interior_srgb(volume, rest_world, tissue):
    from scipy.ndimage import map_coordinates
    color = np.asarray(volume["materials"][tissue-1]["srgb"],np.float32)
    detail = map_coordinates(volume["interior_grain"],voxel_coordinates(volume,rest_world).T,
                             order=1,mode="nearest",prefilter=False,output=np.float32)-128.
    return np.clip((color+detail[:,None])/255.,0.,1.)


def sample_tissues(volume, rest_world, *, soft_only=False):
    index = np.rint(voxel_coordinates(volume,rest_world)).astype(np.int32)
    index = np.clip(index,0,np.array(volume["labels"].shape)-1)
    ix = tuple(index.T)
    tissue = volume["labels"][ix]
    if soft_only and "rigid_id" in volume:
        pulp_id, = [i for i,m in enumerate(volume["materials"],1) if m["name"] == "Pulp"]
        tissue[tissue == volume["rigid_id"]] = pulp_id
    # The reconstructed boundary can be fractionally outside the voxelized rest mesh.
    return np.where(tissue==0,1,tissue).astype(np.uint8), volume["lobes"][ix]


def assign_particle_materials(sim, volume, rest_world):
    tissue, lobes = sample_tissues(volume, rest_world, soft_only=True)
    table = volume["materials"]
    # Only the soft domain is seeded. Nearest-voxel sampling can round an exterior
    # particle to the adjacent rigid cell at a coarse or rotated interface.
    parameters = np.zeros((len(table), 3))
    for i, material in enumerate(table):
        if not material.get("rigid"):
            parameters[i] = material["density"], material["youngs_modulus"], material["poisson_ratio"]
    rho, young, nu = parameters[tissue.astype(int)-1].T
    vol = sim.particles.vol.to_numpy()
    for field, values in [(sim.particles.mass,rho*vol), (sim.particles.mu0,young/(2*(1+nu))),
                          (sim.particles.la0,young*nu/((1+nu)*(1-2*nu)+1e-4))]:
        field.from_numpy(values.astype(np.float32))
    return tissue, lobes
