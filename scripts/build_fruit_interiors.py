"""Construct the apple, peach, cucumber and strawberry tissue assets in OBJ coordinates."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, gaussian_filter1d, map_coordinates
from scipy.spatial import cKDTree

from build_tissue_volumes import ROOT, UPSTREAM, file_hash, gaussian_points, layered_labels, peel_interface, reference_sdf, save_layers


def photographic_detail(source, box, u, v):
    """Sample a de-lit patch of flesh in a continuous, mirrored material-coordinate chart."""
    with Image.open(ROOT/"assets/tissues/references"/(source+".jpg")) as im:
        factor = im.width/600.
        patch = im.crop(tuple(round(k*factor) for k in box)).convert("RGB").resize((192,192), Image.Resampling.LANCZOS)
    pixels = np.asarray(patch, np.float32)
    residual = gaussian_filter(pixels,(1.4,1.4,0)) - gaussian_filter(pixels,(30,30,0))
    coords = 191*(1.-np.abs(np.stack((v,u))%2.-1.))
    return np.stack([map_coordinates(residual[...,i],coords,order=1) for i in range(3)],axis=1)


def apple(q, source_dir):
    source = source_dir/"apple.ply"
    if file_hash(source) != "2ef59241667ef97ea2ff9e506c74df4d1914bedc97144c549388e3592f66e8b7":
        raise ValueError("Unexpected FruitNinja apple checkpoint")
    xyz, dc = gaussian_points(source)
    low, high = np.quantile(xyz,[.0005,.9995],axis=0)
    # The source fruit axis is Z. Pull the skin samples into the flesh along the surface.
    local = q[:,[0,2,1]]
    query = (low+high)/2 + .83*local*(high-low)/2
    tree = cKDTree(xyz)
    nearest = np.concatenate([tree.query(x,workers=4)[1] for x in np.array_split(query,32)])
    rgb = np.clip(.5+.2820947918*dc[nearest],0.,1.)*255
    x,y,z = q.T
    a = np.arctan2(z,x)
    radius = np.hypot(x,z)
    core = (radius/(.26+.045*np.cos(5*a)))**2 + ((y+.015)/.71)**2 < 1.
    seeds = np.zeros(len(q),bool)
    for i in range(5):
        angle = 2*np.pi*i/5
        radial = x*np.cos(angle)+z*np.sin(angle)
        lateral = z*np.cos(angle)-x*np.sin(angle)
        seeds |= ((radial-.165-.13*y)/.060)**2 + (lateral/.048)**2 + ((y-.025)/.145)**2 < 1.
    # Remove skin and dark source pips from the soft flesh. Pips have their own solid geometry.
    pale = np.array([236,224,181])
    weight = np.clip((rgb[:,1]-105)/66,0,1)[:,None]
    flesh = pale + .70*weight*(rgb-pale)
    grain = np.clip((rgb[:,1]-170)*.25,-20,17)
    return [core,seeds], flesh, grain


def peach(q):
    x,y,z = q.T
    a = np.arctan2(z/.27,x/.40)
    grooves = np.cos(10*a+1.6*np.sin(5*y)) * np.sin(16*y+3*np.cos(3*a))
    stone_radius = np.sqrt((x/.38)**2+(y/.54)**2+(z/.26)**2)
    stone = stone_radius < 1.+.075*grooves
    detail = photographic_detail("peach-section",(339,171,385,264),
                                2*x+.43*z+.13*np.sin(3*y), 2*y+.5*z)
    blush = np.exp(-3.6*np.maximum(stone_radius-1,0))[:,None]
    pulp = np.array([247,195,113]) + 1.1*detail - blush*np.array([38,97,61])
    woody = photographic_detail("peach-stone",(298,481,338,508),a/np.pi,3*y)
    grain = woody.mean(1)*.65 + 16*grooves
    return [stone], pulp, grain


def cucumber(q, indices, sdf):
    # Follow the curved cross-section centers of this particular cucumber.
    zindex = indices[:,2]
    counts = np.bincount(zindex,minlength=sdf.shape[2])
    present = np.flatnonzero(counts)
    def section_mean(values):
        avg = np.bincount(zindex,weights=values,minlength=len(counts))[present]/counts[present]
        return gaussian_filter1d(np.interp(np.arange(len(counts)),present,avg),1.8)[zindex]
    x,y,z = q.T
    x = x-section_mean(x)
    y = y-section_mean(y)
    x /= np.maximum(np.sqrt(4*section_mean(x*x)),.14)
    y /= np.maximum(np.sqrt(4*section_mean(y*y)),.14)
    a = np.arctan2(y,x)
    radius = np.hypot(x,y)
    chamber = (radius/(.52+.10*np.cos(3*a+1.1*z)))**2 + (z/.95)**8 < 1
    detail = photographic_detail("cucumber-section",(341,170,372,192),1.5*x+.6*y+2*z,6*z+.4*y)
    inner = np.clip(1.-radius,0.,1.)[:,None]
    pulp = np.array([183,208,141])+inner*np.array([31,15,26])+1.25*detail
    return [chamber],pulp,detail.mean(1)*1.4+7*np.cos(3*a+z)


def strawberry(points):
    # Align the pith with the tilted long axis, from calyx end toward the narrower tip.
    delta = points-points.mean(axis=0)
    _,axes = np.linalg.eigh(np.cov(delta.T))
    if axes[2,2] < 0:
        axes[:,2] *= -1
    local = delta @ axes
    local = 2*(local-local.min(0))/np.ptp(local,axis=0)-1
    x,y,z = local.T
    radius = np.hypot(x,y)
    core = (radius/np.maximum(.24-.125*z,.05))**2+((z+.11)/.82)**2 < 1
    a = np.arctan2(y,x)
    veins = (.5+.5*np.cos(10*a+7*z+1.4*np.sin(3*a)))**4
    detail = photographic_detail("strawberry-section",(326,281,358,351),
                                2.1*x+.8*y+.2*np.sin(4*z),2*z+.75*y)
    center = np.exp(-1.8*radius**2)[:,None]
    pulp = np.array([195,39,44]) + center*np.array([29,53,44]) + detail + (veins*(1.-.45*radius))[:,None]*np.array([10,9,6])
    return [core],pulp,detail.mean(1)+5*np.cos(6*a+2*z)


def stem_partition(path, cap_center, origin, pitch, shape):
    """Close both OBJ groups on their shared socket and sample them on the same XYZ grid."""
    from collections import Counter
    import trimesh
    from fruit_mesh_sdf import mesh_distance_query

    vertices, groups = [], {"Body": [], "Stem": []}
    for line in path.read_text().splitlines():
        w = line.split()
        if not w:
            continue
        if w[0] == "v":
            vertices.append([float(x) for x in w[1:4]])
        elif w[0] == "g":
            group = w[1]
        elif w[0] == "f":
            f = [int(c.split("/")[0])-1 for c in w[1:]]
            groups[group].extend((f[0],a,b) for a,b in zip(f[1:],f[2:]))
    cap = len(vertices)
    vertices.append(cap_center)
    volumes = []
    for faces in groups.values():
        edges = [(a,b) for f in faces for a,b in zip(f, f[1:]+f[:1])]
        counts = Counter(tuple(sorted(edge)) for edge in edges)
        faces.extend((b,a,cap) for a,b in edges if counts[tuple(sorted((a,b)))] == 1)
        part = trimesh.Trimesh(vertices,faces,process=True)
        part.remove_unreferenced_vertices()
        if not part.is_volume:
            raise ValueError(f"Open or inverted component in {path}")
        query = mesh_distance_query(part)
        field = np.empty(shape, np.float32)
        for start in range(0,shape[0],8):
            block = field[start:start+8]
            grid = np.indices(block.shape).reshape(3,-1).T
            field[start:start+8] = query(origin + pitch*(grid+[start,0,0])).reshape(block.shape)
        volumes.append(field)
    return volumes


def build(name, source_dir):
    mesh = UPSTREAM/"assets"/(name+".obj")
    spec = json.loads((ROOT/"assets/tissues"/(name+".json")).read_text())
    if "mesh_variant" in spec:
        import trimesh
        from fruit_mesh_sdf import translated_mesh_sdf
        if file_hash(mesh) != spec["mesh_variant"]["source_sha256"]:
            raise ValueError(f"Unexpected source for {name}")
        mesh = ROOT/"assets/tissues"/spec["mesh_variant"]["mesh"]
        geom = trimesh.load(mesh, force="mesh", process=False, skip_materials=True)
        pack, _, _ = translated_mesh_sdf(geom, mesh, {}, 192 if name=="apple" else 160)
        lo, hi = geom.bounds
        if name == "apple":
            original = trimesh.load(UPSTREAM/"assets/apple.obj", force="mesh", skip_materials=True)
            lo, hi = original.bounds
        sdf, origin, voxel = pack["sdf"].T, pack["origin"], float(pack["voxel_size"])
    else:
        lo,hi,sdf,origin,voxel = reference_sdf(mesh,192 if name in ("apple","cucumber") else 160)
    thickness = spec["geometry"]["peel_thickness_m"]
    body_sdf = sdf
    extra = {}
    if "stem" in spec:
        body_sdf, stalk = stem_partition(mesh, spec["stem"]["base_center_m"], origin, voxel, sdf.shape)
        # The two closed solids meet at the socket. The bisector extends this
        # boundary through the empty exterior without creating coincident shells.
        split = .5*(stalk-body_sdf)/voxel
        extra["stem_interface_vox"] = np.clip(split,-12.,12.).astype(np.float16)
    labels = layered_labels(body_sdf,thickness)
    if "stem" in spec:
        labels[sdf>=0] = 0
        labels[(sdf<0) & (split<0)] = len(spec["materials"])
        labels[(sdf<0) & (labels==0)] = 1
    indices = np.argwhere(labels!=0)
    ix = tuple(indices.T)
    points = origin + voxel*indices
    q = (points-(lo+hi)/2)/((hi-lo)/2)
    if name=="apple":
        masks,colors,grain = apple(q,source_dir)
    elif name=="peach":
        masks,colors,grain = peach(q)
    elif name=="cucumber":
        masks,colors,grain = cucumber(q,indices,sdf)
    else:
        masks,colors,grain = strawberry(points)
    ids = labels[ix]
    inner = ids==2
    for material,mask in enumerate(masks,3):
        ids[inner & mask] = material
    labels[ix] = ids
    if name == "cucumber":
        with np.load(ROOT / "templates/cucumber/anatomy.npz") as anatomy:
            labels = anatomy["labels"]
    rgb = np.full((*sdf.shape,3),spec["materials"][1]["srgb"],np.uint8)
    rgb[ix] = np.clip(np.rint(colors),0,255).astype(np.uint8)
    detail = np.full(sdf.shape,128,np.uint8)
    detail[ix] = np.clip(np.rint(grain)+128,0,255).astype(np.uint8)
    save_layers(spec,mesh,labels,origin,voxel,peel_interface_vox=peel_interface(body_sdf,thickness,voxel),**extra,
                pulp_srgb=rgb,interior_grain=detail)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fruit",choices=("apple","peach","cucumber","strawberry"))
    parser.add_argument("--source-dir",type=Path,default=ROOT/".cache/gaussianfluent")
    args = parser.parse_args()
    for fruit in ([args.fruit] if args.fruit else ("apple","peach","cucumber","strawberry")):
        build(fruit,args.source_dir)
