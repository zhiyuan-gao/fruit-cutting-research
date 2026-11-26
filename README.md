# Fruit cutting physics for Isaac Sim

Generate eight fruits with separate anatomical meshes and simulate their soft tissues with CulinaryCut MPM. The fruit templates link shape, internal anatomy and material parameters. The avocado and peach have fixed pits that stop the blade. Apple and watermelon seeds are individually movable rigid bodies.

**[Watch the eight-fruit demo](demo/fruit-cutting-films.html)**. Download the HTML and open it in a browser. The file contains one 32.25-second, double-speed film with synchronized exterior and interior views.

**[Download the eight USDZ models for Isaac Sim](https://github.com/zhiyuan-gao/fruit-cutting-research/releases/download/v0.1.0/fruit-models-usdz-v0.1.0.zip)**. Extract the ZIP and open any fruit's USDZ directly in Isaac Sim. All textures are embedded. These are the reference samples generated with seed 2026 and index 0.

## Cutting physics experiments

| Experiment | Watch | Inspect the measurements |
| --- | --- | --- |
| Whole cucumber, one cut. Withdraw the knife, push one piece, then bring the faces back into contact. | [Cut and recontact](https://github.com/zhiyuan-gao/fruit-cutting-research/releases/download/v0.2.0/cucumber-cut-and-recontact.mp4) | [Per-frame record](records/cucumber-piece-trial.json) · [Tracked particle crossing the blade](records/cut-alignment.png) · [Continuous collision check](records/cut-alignment-ccd.json) |
| CRESSim-MPM. Give the right piece an outward velocity and compare with an unloaded run. On the native shared grid the right and left pieces moved an additional 8.93 and 8.59 mm. The separate-grid comparison reuses the passive trajectory and tests outward motion only. | [Side-by-side controls, Chinese captions](https://github.com/zhiyuan-gao/fruit-cutting-research/releases/download/v0.2.0/cressim-cucumber-results-zh.mp4) | [Displacements, solver revision and limitations](records/cressim-cucumber.json) · [Build and run](scripts/cressim_trial/README.md) |
| Elastic cucumber-flesh coupon, 6 × 12 × 6 mm. Compare a partial and a through cut, apply a load to one side, then close and reopen the cut. | [Half cut and through cut](https://github.com/zhiyuan-gao/fruit-cutting-research/releases/download/v0.2.0/coupon-cuts.mp4) | [Result figure](records/coupon-results.png) · [26 numerical cases](records/cutting-physics-coupon.json) · [Six-axis wrench CSVs](records/coupon-wrench) |

These simulations use prescribed knife motion. In the coupon, shifting the knife by 0.5 mm relative to the particle rows changed the downstroke work from 2.81 to 8.40 mJ at the same speed and resistance. This sensitivity remains an obstacle to estimating cutting resistance from real force data. Stress-driven cracking and robot calibration are planned. [Research plan and parameters (中文)](CUTTING_PHYSICS_PLAN.md) · [Research handoff (中文)](HANDOFF.md)

Open any of the five USD time series in Isaac Sim and scrub the timeline to inspect recorded states. [Download the replays](https://github.com/zhiyuan-gao/fruit-cutting-research/releases/tag/v0.2.0)

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

## Single-cut cucumber trial in Isaac Sim

Run the numerical experiment on the reference cucumber and open its time-sampled stage in Isaac Sim.

```sh
./python.sh piece_trial.py examples/cucumber/2026_0000 --out outputs/cucumber-piece-trial
./python.sh validate_isaac.py --cut outputs/cucumber-piece-trial/cucumber.usdc
```

The stage shows `/World/Fruit/Whole` during the partial cut. Once the material separates, `Piece_001` and `Piece_002` become visible under `/World/Fruit`. Each Xform follows its mass center and contains points grouped by tissue with stable particle IDs. Scrub the timeline and inspect `mpm:phase` and `mpm:simulationTime` on `/World`. The saved USD plays back computed MPM states. In a running Python simulation, `FruitSimulation(..., track_pieces=True)` exposes `state()["piece_ids"]`. After separation, `set_piece_force(piece_id, [fx, fy, fz])` applies a held total world-space force in N, distributed by particle mass. A zero vector releases the load.

The trial uses one transverse stroke across a 24 mm slot between supports. It traces the lower edge of the registered knife mesh and samples the board in the board's own SDF coordinates. In the initial material configuration, neighbors within 1.25 particle spacings form a graph. Links whose endpoints have acquired opposite cut-side labels cease to connect the material. This detects a remaining bridge at the sampled resolution and preserves piece IDs when cut faces touch again. The graph is evaluated from the MPM cut history, independently of the display.

The trial applies the same velocity decay of 300 per second throughout, then branches from the withdrawn state for a matched unloaded control. It measures each piece's axial displacement, compressive grid contacts and the signed axial separation of originally neighboring particles across the cut. The support, 4 m/s knife speed and distributed load are numerical test conditions. Validation covers a single cut on the reference cucumber. [Reference trial measurements](records/cucumber-piece-trial.json)

## Reproduce the coupon measurements

After preparing the reference samples above, run the following commands from the repository root. The full suite uses the original whole-cucumber recording as a positive control for the blade-crossing detector. The published recording is about 116 MiB.

```sh
./python.sh scripts/prepare_tools.py
mkdir -p outputs/cucumber-piece-trial
archive=outputs/cucumber-piece-trial/cucumber.usdc
if [ ! -f "$archive" ]; then
    curl -fL https://github.com/zhiyuan-gao/fruit-cutting-research/releases/download/v0.2.0/cucumber.usdc -o "$archive.part" && mv "$archive.part" "$archive"
fi
echo "4e141d2a76f088bc2651cb8e28f2cc06703fe203c6e224a62bdfa6336f00d97e  $archive" | sha256sum -c -
./python.sh cut_physics_trial.py full --speed 0.05 --keff 80 --out outputs/cut-physics/full80
scripts/cut_physics/run_trials.sh
```

The single-run command writes the centered cut's force, torque and energy traces. The suite runs the sampling and timestep comparisons, the native CulinaryCut controls, and the cut-piece manipulation tests. It also regenerates the report, figure and coupon USD replay. To compare the published curves directly, use `full80.csv` (centered), `offset80.csv` (0.5 mm lateral shift) and `native-full.csv` (centered CulinaryCut control). All three use 0.05 m/s. The two reference-grid runs use a fixed effective cutting resistance of 80 J/m². Each CSV row ends one averaging window. Times are seconds, depth is millimetres, and the three force and three moment columns give world-frame knife reactions in N and N·m about the wrist reference point. [Published traces](records/coupon-wrench) · [Solver and trial settings](cut_physics_trial.py)

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
