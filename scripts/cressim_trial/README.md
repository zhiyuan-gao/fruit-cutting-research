# CRESSim-MPM reference cucumber trial

The GPU solver runs outside Isaac Sim. The stage exporter writes the measured material-point trajectories for playback. The pinned original engine also ran the upstream `example_cutting` loop with a headless display callback for 310 frames.

The trial uses all 114,517 particles of the seed-2026 cucumber, its tissue densities and elastic parameters, and the 24 mm support slot. The native rectangular CRESSim cutter has a 2 mm contact band on each side. The prescribed downstroke is 4 m/s. The flat cutter, contact model and absence of extra velocity decay differ from the earlier wedge trial. The primary substep is 7.629 microseconds, with a prescribed cutter-pose update every 244.14 microseconds.

For the partial cut, a pair of material patches above the tip is moved outwards by 4 mm each, alongside a zero-displacement hold and an intact cucumber control. Separation is measured using the median axial spacing of originally adjacent particles across the knife plane. Initial spacing is about 2 mm. For the full cut, the loaded side gains an initial outward velocity of 0.6 m/s after withdrawal. The control receives no velocity increment. The full-cut tool is withdrawn to a tip height of 245 mm because material rebounds upwards.

In the native trial the solver continues on one grid. The comparison assigns sides as the finite blade edge reaches the height of each particle. Once the reference neighbor graph breaks into two components, matched replays retain the corresponding particles on separate grids, including their deformation history. This isolates outward motion. It also disables contact between pieces. The passive grid trajectory is reused in the loaded and unloaded compositions.

## Reproduction

The build script pins [CRESSim-MPM commit 09aa500](https://github.com/yafei-ou/CRESSim-MPM/tree/09aa5009b8580351f516b6df7660e87821fc5eb6). The recorded run used CUDA 11.8 with SM89. Set `CRESSIM_CUDA_ARCH` to the architecture of your GPU when building. Use a CUDA development toolkit with a compatible GPU. Host Python requires NumPy, SciPy and Pillow. The movie exporters use FFmpeg and the DejaVu Sans (English) or Noto Sans CJK (Chinese) system fonts. Run from the fruit-cutting-research root after generating the reference cucumber. Replace `python3` with the corresponding Python executable if necessary.

```bash
scripts/cressim_trial/build.sh
export OPENBLAS_NUM_THREADS=1
trial=scripts/cressim_trial/run.py
raw=.cache/cressim-run
CRESSIM_SUBDIVIDE=2 CRESSIM_RAW_OUT=$raw/refined python3 "$trial" half_load half_free sham_load sham_free native_load native_free
CRESSIM_SUBDIVIDE=2 CRESSIM_SPLIT_AT_DEPTH=1 CRESSIM_RAW_OUT=$raw/refined-early python3 "$trial" split_free
CRESSIM_SUBDIVIDE=2 CRESSIM_SPLIT_AT_DEPTH=1 CRESSIM_ALL_FRAMES=1 CRESSIM_RAW_OUT=$raw/usd-frames python3 "$trial" half_load split_load split_passive
CRESSIM_RAW_OUT=$raw/clear-coarse python3 "$trial" native_free
CRESSIM_SUBDIVIDE=4 CRESSIM_RAW_OUT=$raw/refined4 python3 "$trial" native_free native_load
mkdir -p outputs/cressim-cucumber
python3 scripts/cressim_trial/report.py
python3 scripts/cressim_trial/movie.py
python3 scripts/cressim_trial/results_video_zh.py
./python.sh scripts/cressim_trial/export_stage.py
```

The scene-boundary and tool-clearance guards stop invalid runs. GPU repeats can differ. One repeat reached the spatial domain boundary and was excluded. Baseline physical motion was sensitive to substep size. Compare the loaded and unloaded branches and the timestep check in `results.json` before interpreting a single video. The native stage contains five full-particle keyframes from the measured loaded replay. The other two stages contain every recorded frame of the corresponding all-particle replays, with the underlying physical time in `/World.trial:timeSeconds`.
