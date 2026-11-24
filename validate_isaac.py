from pathlib import Path
import json
import argparse
import sys

parser = argparse.ArgumentParser(
    description="Open all sampled USDZ files in Isaac Sim and check their geometry and textures"
)
parser.add_argument(
    "--cut", type=Path, help="Also verify a time-sampled stage from simulate.py --stage"
)
args = parser.parse_args()
sys.argv = sys.argv[:1]
from isaacsim import SimulationApp

app = SimulationApp(
    {
        "headless": True,
        "hide_ui": True,
        "multi_gpu": False,
        "width": 320,
        "height": 240,
        "limit_cpu_threads": 8,
        "extra_args": ["--/app/extensions/registryEnabled=false"],
    },
    experience="/isaac-sim/apps/isaacsim.exp.base.python.kit",
)
import omni.usd
from pxr import UsdGeom, UsdShade, UsdPhysics

root = Path(__file__).resolve().parent
ctx = omni.usd.get_context()
results = {}


def check():
    files = sorted((root / "examples").glob("*/*/fruit.usdz"))
    assert {f.parents[1].name for f in files} == {
        p.parent.name for p in (root / "templates").glob("*/template.json")
    }
    for f in files:
        assert ctx.open_stage(str(f)), f
        for i in range(4):
            app.update()
        s = ctx.get_stage()
        meshes = [str(p.GetPath()) for p in s.Traverse() if p.IsA(UsdGeom.Mesh)]
        textures = [
            UsdShade.Shader(p).GetInput("file").Get().resolvedPath
            for p in s.Traverse()
            if p.IsA(UsdShade.Shader)
            and UsdShade.Shader(p).GetIdAttr().Get() == "UsdUVTexture"
        ]
        assert meshes and textures and all(textures)
        results[str(f.relative_to(root))] = {
            "mesh_nodes": len(meshes),
            "resolved_textures": len(textures),
            "movable_seeds": sum(
                p.HasAPI(UsdPhysics.RigidBodyAPI) for p in s.Traverse()
            ),
        }
    if args.cut:
        assert ctx.open_stage(str(args.cut.resolve()))
        stage = ctx.get_stage()
        times = (
            stage.GetPrimAtPath("/World")
            .GetAttribute("mpm:simulationTime")
            .GetTimeSamples()
        )
        assert len(times) >= 2 and times[-1] == stage.GetEndTimeCode()
        assert stage.GetPrimAtPath("/World/Knife") and any(
            p.IsA(UsdGeom.Points) for p in stage.Traverse()
        )
        results[str(args.cut)] = {"frames": len(times)}
    print(json.dumps(results, indent=2), flush=True)


check()
app.close()
