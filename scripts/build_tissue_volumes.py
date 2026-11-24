"""Build asset-local tissue labels from the fruit surface or registered Gaussian interiors."""
import argparse
import hashlib
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "third_party/EEF-Cutting-Simulation"
SOURCES = {
    "orange": ("orange.ply", "42f0d6acaab6820fd013ab9a7d8471f7c3e5582cff811924ae3c0f423054886e"),
    "melon": ("watermelon.ply", "fc5c4871b5f0b78714a5eafe4f174c3d874a29b541fb0afd9fff1a556d55c834"),
}


def file_hash(path):
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def gaussian_points(path):
    with path.open("rb") as stream:
        header = []
        while True:
            line = stream.readline()
            if not line or len(header) > 200:
                raise ValueError("Invalid Gaussian PLY header")
            if line.strip() == b"end_header":
                break
            header.append(line.decode().strip())
        offset = stream.tell()
    fields = [line.split()[-1] for line in header if line.startswith("property float ")]
    data = np.memmap(path, dtype="<f4", offset=offset, mode="r").reshape(-1, len(fields))
    keep = data[:, fields.index("opacity")] > 0
    return data[keep, :3], data[keep, fields.index("f_dc_0"):fields.index("f_dc_0")+3]


def reference_sdf(mesh, resolution=192):
    vertices = np.array([line.split()[1:4] for line in mesh.read_text().splitlines() if line.startswith("v ")], float)
    lo, hi = vertices.min(0), vertices.max(0)
    expected_step = (hi-lo+.02).max()/resolution
    for path in sorted((UPSTREAM/"cache_sdf").glob("cuttingmesh.*.npz")):
        with np.load(path) as data:
            if (np.allclose(data["asset_bounds_local_lb"], lo) and np.allclose(data["asset_bounds_local_ub"], hi)
                    and np.allclose(data["T_asset_to_world"], np.eye(4))
                    and np.isclose(data["voxel_size"], expected_step)):
                return lo, hi, data["sdf"].T.copy(), data["origin"], float(data["voxel_size"])
    raise ValueError(f"No matching native SDF for {mesh}")


def layered_labels(sdf, thickness):
    """XYZ labels: empty exterior, skin measured inward from the surface, and flesh."""
    labels = np.zeros(sdf.shape, dtype=np.uint8)
    labels[sdf < 0] = 1
    labels[sdf < -thickness] = 2
    return labels


def peel_interface(sdf, thickness, voxel):
    """Signed distance in source voxels to the same boundary used to assign tissue IDs."""
    return np.clip((sdf + thickness)/voxel, -8., 8.).astype(np.float16)


def melon_pulp_interface(inside, voxel, smoothing_m):
    """Smooth the rendered rind/pulp boundary at rest, preserving its enclosed volume."""
    from scipy.ndimage import gaussian_filter

    sigma = smoothing_m/voxel
    occupancy = gaussian_filter(inside.astype(np.float32),sigma,mode="constant")
    # Choose the isovalue by enclosed voxel count. A fixed 0.5 threshold
    # shrinks convex surfaces as the smoothing radius increases.
    boundary = occupancy.size-np.count_nonzero(inside)
    ranks = np.partition(occupancy.ravel(),(boundary-1,boundary))
    level = 0.5*(ranks[boundary-1]+ranks[boundary])
    # Invert the slope of a Gaussian-blurred step at the interface to put
    # the field in voxel units, as expected by the live mesh extractor.
    distance = (level-occupancy)*(np.sqrt(2*np.pi)*sigma)
    return np.clip(distance,-8.,8.).astype(np.float16)


def save_layers(spec, mesh, labels, origin, voxel, *, lobes=None, **geometry):
    target = ROOT/"assets/tissues"/spec["volume"]
    if lobes is None:
        lobes = np.zeros_like(labels)
    with NamedTemporaryFile(dir=target.parent, delete=False) as stream:
        pending = Path(stream.name)
        try:
            np.savez_compressed(stream, labels=labels, lobes=lobes, origin=origin,
                                voxel=voxel, mesh_sha256=file_hash(mesh), **geometry)
        except BaseException:
            pending.unlink(missing_ok=True)
            raise
    pending.chmod(0o644)
    pending.replace(target)
    print(target.name, np.bincount(labels.ravel()).tolist(), flush=True)


def banana_pulp_distance(sdf, origin, voxel, geometry):
    """Round off the flesh at the OBJ's body-to-stalk ring in asset coordinates."""
    direction = np.asarray(geometry["stem_direction"], dtype=float)
    direction /= np.linalg.norm(direction)
    center = geometry["stem_base_center_m"]
    axes = np.ogrid[tuple(slice(n) for n in sdf.shape)]
    toward_stalk = sum((origin[i] + axis * voxel - center[i]) * direction[i]
                       for i, axis in enumerate(axes))
    under_peel = sdf + geometry["peel_thickness_m"]
    # A smooth intersection keeps the tip rounded and leaves the distant body untouched.
    blend = geometry["pulp_tip_blend_m"]
    overlap = np.maximum(blend - np.abs(under_peel - toward_stalk), 0.)
    return np.maximum(under_peel, toward_stalk) + overlap**2 / (4. * blend)


def banana_pulp_color(pulp_distance, voxel, base_color):
    from PIL import Image
    from scipy.ndimage import gaussian_filter, gaussian_filter1d, map_coordinates

    photo = ROOT/"assets/tissues/references/banana-high-detail.jpg"
    if file_hash(photo) != "a2bbb866efd6e1c8be53bdc7094a7a3c9648fc713d2a42c62809277745ba5d32":
        raise ValueError("Unexpected banana flesh reference photograph")
    with Image.open(photo) as image:
        rgb = np.asarray(image.convert("RGB"), np.float32)
    # Four points inside the flesh, expressed on an 850-pixel-wide preview.
    # Rectify this patch so its long fibers run along the horizontal axis.
    u, v = np.meshgrid(np.linspace(0.,1.,600), np.linspace(0.,1.,256))
    x = (1-v)*(242+412*u) + v*(252+455*u)
    y = (1-v)*(323-219*u) + v*(541-128*u)
    photo_coords = np.stack((y,x)) * (rgb.shape[1]/850.)
    patch = np.stack([map_coordinates(rgb[...,c],photo_coords,order=1) for c in range(3)],axis=-1)
    detail = gaussian_filter(patch,(1.6,3.,0)) - gaussian_filter(patch,(23.,64.,0))

    # Trace the curved center of the flesh along the OBJ's longitudinal Z axis.
    x, y, _ = np.ogrid[tuple(slice(n) for n in pulp_distance.shape)]
    inside = pulp_distance < 0.
    area = inside.sum(axis=(0,1))
    valid = np.flatnonzero(area)
    levels = np.arange(len(area))
    def center(axis):
        sums = (inside*axis).sum(axis=(0,1))
        return gaussian_filter1d(np.interp(levels,valid,sums[valid]/area[valid]),1.5)
    cx, cy = center(x), center(y)
    along = np.r_[0.,np.cumsum(np.sqrt(1.+np.diff(cx)**2+np.diff(cy)**2))]*voxel
    theta = np.arctan2(y-cy, x-cx)
    axial = along/.105
    u = axial + .09*np.sin(theta+axial)
    v = theta/np.pi + .11*np.sin(1.7*axial) + .05*np.sin(3.*theta+.7*axial)
    # Mirror the crop at its edges so adjacent patches meet continuously.
    coords = np.stack(((1.-np.abs(v%2.-1.))*(patch.shape[0]-1),
                       (1.-np.abs(u%2.-1.))*(patch.shape[1]-1)))
    fibers = np.stack([map_coordinates(detail[...,c],coords,order=1) for c in range(3)],axis=-1)
    shell = np.exp(-np.maximum(-pulp_distance,0.)/.009)
    radius = np.hypot(x-cx,y-cy)*voxel
    strength = (1.25*shell + .12)*np.minimum(radius/.010,1.)
    color = base_color + strength[...,None]*fibers - (1.-shell[...,None])*np.array([2.,4.,9.])
    return np.rint(color).clip(0,255).astype(np.uint8)


def build_banana():
    mesh = UPSTREAM/"assets/banana.obj"
    spec = json.loads((ROOT/"assets/tissues/banana.json").read_text())
    _, _, sdf, origin, voxel = reference_sdf(mesh, resolution=160)
    distance = banana_pulp_distance(sdf, origin, voxel, spec["geometry"])
    labels = (sdf < 0).astype(np.uint8)
    labels[distance < 0] = 2
    pulp = next(m for m in spec["materials"] if m["name"] == "Pulp")
    color = banana_pulp_color(distance, voxel, np.asarray(pulp["srgb"]))
    save_layers(spec, mesh, labels, origin, voxel,
                peel_interface_vox=np.clip(distance/voxel,-8.,8.).astype(np.float16), pulp_srgb=color)


def melon_seeds(shape, indices, dark, rgb, rind, voxel, geometry):
    """Fit one bounded pip at each well-separated peak in captured dark flesh."""
    from scipy.ndimage import distance_transform_edt, gaussian_filter, maximum_filter

    ix = tuple(indices.T)
    pulp = np.zeros(shape,bool)
    pulp[ix] = ~rind
    strength = np.zeros(shape,np.float32)
    strength[ix] = dark * np.clip((.48-rgb.mean(axis=1))/.36,0.,1.) * ~rind
    axes = np.asarray(geometry["seed_dimensions_m"])/(2*voxel)
    min_spacing = geometry["min_seed_center_spacing_m"]/voxel
    # Leave pulp both between pips and between each pip and the rind.
    if min_spacing <= 2*axes.max()+2:
        raise ValueError("Seed spacing must leave a resolvable gap")
    score = gaussian_filter(strength,sigma=.0026/voxel)
    safe = distance_transform_edt(pulp) > axes.max()+1
    peaks = np.argwhere(safe & (strength>0) & (score>.30) & (score==maximum_filter(score,size=9)))
    seeds = np.zeros(shape,bool)
    centers = []
    half_window = int(np.ceil(axes.max()+1))
    for i in np.argsort(-score[tuple(peaks.T)],kind="stable"):
        center = peaks[i]
        if any(np.linalg.norm(center-other)<min_spacing for other in centers):
            continue
        window = tuple(slice(c-half_window,c+half_window+1) for c in center)
        offsets = np.moveaxis(np.indices(strength[window].shape),0,-1)-half_window
        xyz = offsets.reshape(-1,3)
        weights = strength[window].ravel()*np.exp(-np.sum(xyz**2,axis=1)/(2*axes[1]**2))
        # The local dark patch determines the orientation; its streak length
        # does not determine the seed size.
        _,basis = np.linalg.eigh(np.cov(xyz.T,aweights=weights,ddof=0))
        basis = basis[:,::-1]
        if np.dot(basis[:,0],center-(np.array(shape)-1)/2) < 0:
            basis[:,0] *= -1
        along,across,thick = np.moveaxis(offsets @ basis / axes,-1,0)
        taper = 1.-.21*along
        pip = along**2+(across**2+thick**2)/taper**2 < 1.
        seeds[window] |= pip
        centers.append(center)
    if not centers:
        raise ValueError("No seed centers in registered watermelon flesh")
    return seeds[ix],np.asarray(centers)


def build(name, source_dir):
    source_name, digest = SOURCES[name]
    spec = json.loads((ROOT/"assets/tissues"/(name+".json")).read_text())
    source = source_dir/source_name
    if file_hash(source) != digest:
        raise ValueError(f"Unexpected checkpoint for {source}")
    mesh = UPSTREAM/"assets"/(name+".obj")
    lo, hi, sdf, origin, voxel = reference_sdf(mesh)
    indices = np.argwhere(sdf < 0)
    local = (origin + voxel*indices - (lo+hi)/2) / ((hi-lo)/2)
    depth = -sdf[tuple(indices.T)] / ((hi-lo).min()/2)
    xyz, dc = gaussian_points(source)
    # Align source Y (watermelon's long axis) with the target melon's long Z axis.
    # Source orange and target orange are both upright along Y.
    low, high = np.quantile(xyz, [.0005, .9995], axis=0)
    axis = (0, 2, 1) if name == "melon" else (0, 1, 2)
    query = ((low+high)/2 + local[:,axis]*(high-low)/2).astype(np.float32)
    tree = cKDTree(xyz)
    nearest = np.concatenate([tree.query(chunk,workers=4)[1] for chunk in np.array_split(query, 32)])
    rgb = np.clip(.5 + .2820947918*dc[nearest], 0, 1)
    lobes = np.zeros(sdf.shape,np.uint8)
    extra = {}
    if name == "melon":
        # The official example expands dark DC candidates by 0.03 in source units.
        dark = ((dc > -2.5) & (dc < -.5)).all(axis=1)
        seed_tree = cKDTree(xyz[dark])
        distances = np.concatenate([seed_tree.query(chunk,distance_upper_bound=.03,workers=4)[0]
                                    for chunk in np.array_split(query,32)])
        rind = (depth < .085) | ((depth < .24) & (rgb[:,1] > .81*rgb[:,0]))
        tissue = np.full(len(local),2,np.uint8)
        candidates = np.isfinite(distances) & (rgb.max(axis=1) < .57)
        seeds,centers = melon_seeds(sdf.shape,indices,candidates,rgb,rind,voxel,spec["geometry"])
        tissue[seeds] = 3
        extra["seed_centers_local_m"] = origin + centers*voxel
        tissue[rind] = 1
        # Seed cavities and newly cut surfaces are constructed separately.
        inside = np.zeros(sdf.shape,bool)
        inside[tuple(indices.T)] = ~rind
        extra["peel_interface_vox"] = melon_pulp_interface(
            inside,voxel,spec["geometry"]["rind_display_smoothing_m"])
        # Extend pulp colors across the inner rind and seed boundaries for stable
        # interpolation on either side of the reconstructed surfaces.
        from scipy.ndimage import distance_transform_edt
        pulp = np.zeros(sdf.shape,bool)
        pulp[tuple(indices.T)] = tissue == 2
        # The modeled rind includes the white mesocarp. On its inner boundary
        # select pink parenchyma from just below the white band in the capture.
        source_rgb = np.clip(.5+.2820947918*dc,0.,1.)
        red = ((source_rgb[:,0]>.48) & (source_rgb[:,0]>1.45*source_rgb[:,1])
               & (source_rgb[:,0]>1.35*source_rgb[:,2]))
        red_tree = cKDTree(xyz[red])
        p = local[tissue==2]
        edge = np.clip((np.linalg.norm(p,axis=1)-.55)/.38,0.,1.)
        inward = 1.-.23*edge**2*(3.-2.*edge)
        red_query = (low+high)/2 + p[:,axis]*inward[:,None]*(high-low)/2
        matched = np.concatenate([red_tree.query(chunk,workers=4)[1] for chunk in np.array_split(red_query,32)])
        flesh_color = source_rgb[red][matched]
        field = np.empty(sdf.shape+(3,),np.uint8)
        field[pulp] = np.rint(255*(.86*flesh_color+.14*np.array([.91,.33,.32])))
        nearest_pulp = distance_transform_edt(~pulp,return_distances=False,return_indices=True)
        extra["pulp_srgb"] = field[tuple(nearest_pulp)]
        grain = np.full(sdf.shape,128,np.uint8)
        grain[tuple(indices.T)] = np.clip(128+38*(rgb.mean(axis=1)-.35),0,255).astype(np.uint8)
        extra["interior_grain"] = grain
    else:
        rind_depth = .076
        # The registered ten-segment edition fills the former seed regions with their surrounding flesh.
        with np.load(ROOT / "templates/orange/anatomy.npz") as anatomy:
            tissue = anatomy["labels"][tuple(indices.T)]
            lobe = anatomy["lobes"][tuple(indices.T)]
        extra["peel_interface_vox"] = peel_interface(sdf, rind_depth*(hi-lo).min()/2, voxel)
        lobes[tuple(indices.T)] = lobe
        # The reference includes white pith under the rind. For the outside of a
        # peeled wedge, sample just underneath it. Keep deeper material in register.
        edge = np.clip((np.linalg.norm(local,axis=1)-.68)/.28, 0., 1.)
        inward = 1.-.10*edge**2*(3.-2.*edge)
        queries = (low+high)/2 + local*inward[:,None]*(high-low)/2
        matched = np.concatenate([tree.query(chunk,workers=4)[1] for chunk in np.array_split(queries,32)])
        flesh = np.clip(.5 + .2820947918*dc[matched], 0., 1.)
        # The captured colors include lighting. Temper saturated patches for relighting.
        flesh = .75*flesh + .25*np.array([.92,.54,.25])
        # The dark central cavity in the reference has no counterpart in these solid wedges.
        # Blend it into the modeled pith along the fruit axis.
        pith = np.clip(1.-np.hypot(local[:,0],local[:,2])/.16,0.,1.)
        pith = (pith**2*(3.-2.*pith))[:,None]
        flesh = (1.-pith)*flesh + pith*np.array([.93,.78,.56])
        colors = {m["name"]:np.asarray(m["srgb"])/255. for m in spec["materials"]}
        membrane = tissue == 3
        flesh[membrane] = .70*flesh[membrane] + .30*colors["Membrane"]
        field = np.empty(sdf.shape+(3,),np.uint8)
        field[...] = np.rint(255*colors["Pulp"])
        field[tuple(indices.T)] = np.rint(255*flesh).astype(np.uint8)
        extra["pulp_srgb"] = field
    labels = np.zeros(sdf.shape,np.uint8)
    labels[tuple(indices.T)] = tissue
    save_layers(spec,mesh,labels,origin,voxel,lobes=lobes,source_sha256=digest,**extra)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir",type=Path, nargs="?", help="Gaussian checkpoints for orange and melon")
    parser.add_argument("--fruit", choices=(*SOURCES, "banana"))
    args = parser.parse_args()
    if args.fruit == "banana":
        build_banana()
    else:
        if args.source_dir is None:
            parser.error("orange and melon require a checkpoint directory")
        for fruit in ([args.fruit] if args.fruit else SOURCES):
            build(fruit,args.source_dir)
