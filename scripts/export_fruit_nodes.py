"""Write a self-contained rest-pose USDZ from the live Isaac fruit geometry."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

from pxr import Sdf, Usd, UsdGeom, UsdShade

from mpm_cutting import CuttingEnvironment
from mpm_cutting.isaac_view import IsaacCuttingView


def export(config, tissue_model, package):
    package = package.resolve()
    package.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=package.parent) as temp, CuttingEnvironment() as env:
        fruit = env.reset(config, tissue_model=tissue_model)
        stage_path = Path(temp) / f"{package.stem}.usdc"
        stage = Usd.Stage.CreateNew(str(stage_path))
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        world = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(world.GetPrim())
        view = IsaacCuttingView(stage, env.scene)
        view.update(fruit, env.get_knife_state())
        surface = view.surface_view
        if not surface.pulp:
            raise ValueError("Choose a tissue model with peel and pulp surface layers")
        for path in ("/World/CulinaryCut/Knife", "/World/CulinaryCut/Board"):
            stage.RemovePrim(path)
        surface.root.RemoveProperty("mpm:positions")
        texture = surface.surface.texture
        if texture is not None:
            shader = UsdShade.Shader.Get(
                stage, surface.peel.GetPath().AppendChild("Texture")
            )
            shader.GetInput("file").Set(Sdf.AssetPath("textures/" + texture.name))
        stage.GetRootLayer().Save()
        staged = Path(temp) / package.name
        writer = Usd.ZipFileWriter.CreateNew(str(staged))
        if not writer.AddFile(str(stage_path), stage_path.name):
            raise OSError(f"Could not add the root layer to {package}")
        if texture is not None and not writer.AddFile(
            str(texture), "textures/" + texture.name
        ):
            raise OSError(f"Could not add {texture} to {package}")
        if not writer.Save():
            raise OSError(f"Could not save {package}")
        staged.replace(package)
    print(package, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("tissue_model", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    export(args.config, args.tissue_model, args.output)
