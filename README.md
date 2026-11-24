# Fruit cutting with CulinaryCut and Isaac Sim

Generate eight fruits with separate anatomical meshes and simulate their soft tissues with CulinaryCut MPM. The fruit templates link shape, internal anatomy and material parameters. The avocado and peach have fixed pits that stop the blade. Apple and watermelon seeds are individually movable rigid bodies.

**[Watch the eight-fruit demo](demo/fruit-cutting-films.html)**. Download the HTML and open it in a browser. The file contains one 32.25-second, double-speed film with synchronized exterior and interior views.

**[Download the eight USDZ models for Isaac Sim](https://github.com/zhiyuan-gao/fruit-cutting-research/releases/download/v0.1.0/fruit-models-usdz-v0.1.0.zip)**. Extract the ZIP and open any fruit's USDZ directly in Isaac Sim. All textures are embedded. These are the reference samples generated with seed 2026 and index 0.

## Reproduce the reference samples

The commands below target Linux x86-64 with an NVIDIA GPU, NVIDIA Container Toolkit, Docker, Python 3 and curl. The container uses Isaac Sim 5.1.0. Review the [Isaac Sim container requirements and terms](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_container.html) before running it.

Run these commands in the repository root. The fetcher downloads each public input from its recorded URL and checks its SHA-256. It reuses already verified files. The largest inputs are the three reference point clouds for the interior colors.

```sh
python3 scripts/fetch_sources.py
python3 prepare_bullet.py
docker build -t fruit-cutting-research:local .
./python.sh scripts/prepare_models.py
./python.sh generate.py --fruit all --seed 2026 --count 1 --out examples
./python.sh validate.py --reference
```

Open `examples/<fruit>/2026_0000/fruit.usdz` in Isaac Sim to inspect the textured meshes, material bindings and individual rigid bodies. The default banana sample is green. Each sample directory also contains `mpm.npz` (registered soft particles and rigid geometry) and `sample.json` (maturity, shape, material values and USD prim paths). Template USDZ files are built locally under `templates/<fruit>/base.usdz`.

`scripts/prepare_models.py --fruit banana` builds a single template after fetching the inputs. To generate more samples, choose a new output directory or use `--start` for new indices. For example, the following command samples three yellow bananas.

```sh
./python.sh generate.py --fruit banana --stage yellow --count 3 --seed 42 --out outputs/yellow
```

## Run a cut and inspect the solver state

```sh
./python.sh simulate.py examples/orange/2026_0000 --speed 4 --report outputs/orange.json --stage outputs/orange.usdc
./python.sh validate_isaac.py --cut outputs/orange.usdc
```

Open `outputs/orange.usdc` in Isaac Sim and scrub the timeline to see MPM particles grouped by tissue, the actual blade pose, and the rigid bodies for fruits that contain them. Each timeline frame represents one recorded state. The `mpm:simulationTime` attribute on `/World` gives physical time in seconds. Omit `--stage` to run just the simulation. The 4 m/s command reproduces the reference test speed. It is a numerical test setting.

The default straight stroke places the grip beyond the fruit and stops at the board or a fixed pit. For custom control, construct `FruitSimulation` from a generated sample, use `command(position, velocity, quaternion, angular)` and advance with `step(ticks)`. `state()` returns soft particle positions, velocities, tissue IDs, segment IDs, the knife pose and the transforms of the rigid prims. Positions are in metres in USD Z-up coordinates. Quaternions use wxyz. `src/mpm_cutting` also contains the upstream solver adapter and its Isaac surface view.

## Model contents

| Fruit | Modeled structures |
|---|---|
| Apple | Peel, flesh, soft core, five moving seeds and stalk |
| Avocado | Peel, flesh and fixed pit |
| Banana | Peel including the stalk, and flesh. Green and yellow stages have different flesh stiffness |
| Cucumber | Peel, flesh and a continuous soft seed chamber |
| Orange | Peel, ten segments and segment membranes |
| Peach | Peel, flesh, fixed stone and stalk |
| Strawberry | Outer layer, flesh and central pith. Surface achenes appear in the texture |
| Watermelon (`melon`) | Green and white rind layers, flesh and 54 moving seeds. The rind layers share a display mesh |

Each `templates/<fruit>/template.json` records sampling bounds, units, stage overrides and the basis for each material value. `anatomy.npz` stores the registered tissue and lobe labels and their voxel-to-USD transform. Shapes and tissues share a smooth deformation during sampling. The [measurement table](records/fruit-tissue-measurements.csv) records reported values, test conditions and paper links. Tissue stiffness and density include measurements and engineering estimates. Seed adhesion, cutting force and tissue-specific fracture toughness still require experimental calibration.

## Validation

`validate.py --reference` compares the eight reference samples against the [recorded particle, rigid-body and texture hashes](records/reference-samples.json). It also checks USD texture resolution, component topology, physics assignments and repeatability. Run the contact probes on a generated sample with `./python.sh validate.py --physics examples/avocado/2026_0000` or `./python.sh validate.py --physics examples/apple/2026_0000`. Isaac Sim stage checks run with `./python.sh validate_isaac.py`.

The [fixed-pit cut](records/avocado-through-to-pit.json), [orange stroke](records/orange-through-cut.json), [apple seed trajectory](records/apple-seed-track.json) and [handle clearance](records/gallery-handle-clearance.json) are numerical records from the film's reference runs. GPU contact trajectories and small surface or shading details can vary with the numerical stack.

## Upstream sources

- **CulinaryCut** supplies the baseline solver, seven fruit exteriors, cutting tool and board. The local adapter applies the swept-fracture patch to a temporary copy at runtime. [Pinned EEF-Cutting-Simulation revision](https://github.com/Chayoso/EEF-Cutting-Simulation/tree/c18304aa83744da0021472c0a63430005a4fc28c)
- **EmbodiedSWE** supplies the whole avocado and peel albedo. Its preparation script repairs one missing triangle, scales the fruit by 1.4 and generates the ellipsoid pit used in the RoboNinja demo. [Pinned avocado asset](https://huggingface.co/datasets/EmbodiedSWE/robobench-assets/tree/62fab5b70ccb0103c60c4ca39236abae2cc8e900/robobench/suites/packing/assets/clear_organic_objects/avocado)
- **GaussianFluent and FruitNinja** supply the registered watermelon, apple and orange reference point clouds used by the tissue preparation scripts. [Pinned model files](https://huggingface.co/hbpencil01/GaussianFluent/tree/bb5817eb88b1eff64e4271b445a22b9fc42c0925/model)
- The interior detail recipes use photographs by Paolo Neo (banana), USDA (peach flesh), Patrick Tregenza (peach stone), Judgefloro (cucumber) and 5snake5 (strawberry). [Banana](https://commons.wikimedia.org/wiki/File:Banana_high_detail.jpg) · [Peach flesh](https://commons.wikimedia.org/wiki/File:USDA_Eat_Healthy_Food_Group_Gallery_serving_size_half_large_peach.jpg) · [Peach stone](https://commons.wikimedia.org/wiki/File:Flavorcrest_peaches.jpg) · [Cucumber](https://commons.wikimedia.org/wiki/File:21Cucumber_slices.jpg) · [Strawberry](https://commons.wikimedia.org/wiki/File:Strawberry_Half.jpg)

The [source manifest](sources.lock.json) records download URLs and SHA-256 digests. The repository contains the code, research data and finished HTML. Each model in the release archive has an individual checksum. The archive includes its source credits and upstream notices.
