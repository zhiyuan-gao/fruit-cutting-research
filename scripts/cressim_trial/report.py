"""Reduce the loaded/control GPU replays. Distances refer to material points."""

import json
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parents[2]
cache = root / ".cache/cressim-run"
out = root / "outputs/cressim-cucumber"
sample = root / "examples/cucumber/2026_0000"
s = np.load(sample / "mpm.npz")
rest = s["x"].astype(float)
meta = json.loads((sample / "sample.json").read_text())
rho = [meta["materials"][k]["density_kg_m3"] for k in meta["material_order"]]
mass = s["volume_m3"] * np.array(rho)[s["tissue"] - 1]
plane = (rest[:, 1].max() + rest[:, 1].min()) / 2


def load(folder, name):
    return np.load(cache / folder / (name + ".npz"))


def gap(d, key):
    low, high = d["edges"]
    g = (d[key][high, 1] - d[key][low, 1]) * 1000
    return {label: float(np.median(g[d[label]])) for label in ("upper", "lower")}


def increment(d, mask):
    return float(
        np.average(d["end"][mask, 1] - d["withdrawn"][mask, 1], weights=mass[mask])
        * 1000
    )


def response(a, b, sign):
    mask = (a["side"] == sign) & (b["side"] == sign)
    return increment(a, mask) - increment(b, mask)


half = {}
for condition in ["half_load", "half_free", "sham_load", "sham_free"]:
    d = load("refined", condition)
    half[condition] = {key: gap(d, key) for key in ("full_depth", "withdrawn", "end")}
    sign = np.sign(rest[:, 1] - plane)
    grip = (
        (abs(rest[:, 1] - plane) > 0.012)
        & (abs(rest[:, 1] - plane) < 0.035)
        & (rest[:, 2] > 0.07856)
    )
    half[condition]["grip_median_shift_mm"] = [
        float(np.median((d["end"] - d["withdrawn"])[(sign == v) & grip, 1]) * 1000)
        for v in (-1, 1)
    ]
for name, control in [("half_load", "half_free"), ("sham_load", "sham_free")]:
    half[name]["extra_vs_hold_mm"] = {
        k: half[name]["end"][k] - half[control]["end"][k] for k in ("upper", "lower")
    }
native = load("refined", "native_load")
free = load("refined", "native_free")
right = load("usd-frames", "split_load")
right_free = load("refined-early", "split_free")
left = load("usd-frames", "split_passive")
assert np.array_equal(left["keep"], ~right["keep"]) and np.array_equal(
    left["side"], right["side"]
)
assert all(
    d["marked"] == len(rest) and d["components"] == 2
    for d in (native, free, right, right_free, left)
)
events = json.loads(str(native["crossings"]))
pair = [71185, 71186]


def pair_dist(x):
    return float(np.linalg.norm(x[pair[1]] - x[pair[0]]) * 1000)


sensitivity = {}
for folder in ("clear-coarse", "refined", "refined4"):
    d = load(folder, "native_free")
    sensitivity[
        {"clear-coarse": "15.259", "refined": "7.629", "refined4": "3.815"}[folder]
    ] = dict(
        half_depth_neighbor_spacing_mm=gap(d, "half_depth"),
        endpoint_neighbor_spacing_mm=gap(d, "end"),
        endpoint_max_height_mm=float(d["end"][:, 2].max() * 1000),
    )
report = dict(
    source="https://github.com/yafei-ou/CRESSim-MPM/tree/09aa5009b8580351f516b6df7660e87821fc5eb6",
    hardware="RTX 4070, CUDA toolkit 11.8, SM89, upstream GPU MLS-MPM",
    official_smoke="Official example_cutting physics loop with headless display callback, 310 frames, 10000 finite particles.",
    sample="examples/cucumber/2026_0000",
    n=len(rest),
    mass_kg=float(mass.sum()),
    materials=meta["materials"],
    setup=dict(
        grid_mm=4,
        substep_us=1e6 / 131072,
        kinematic_update_us=1e6 / 4096,
        trial_units="positions m, seconds; CRESSim internally cm, g, s",
        support_slot_mm=24,
        start_tip_m=0.12369,
        half_depth_tip_m=0.06656,
        through_tip_m=0.021,
        half_withdraw_tip_m=0.13869,
        full_withdraw_tip_m=0.245,
        blade="native quad slicer 160 x 61.4 mm, fatten 2 mm on each side, friction 0",
        speed_m_s=4,
        gravity_m_s2=9.81,
        support_friction=0.35,
        test_duration_ms=31.25,
        half_open="Material patches 12-35 mm to either side above z=78.56 mm, prescribed axial movement 4 mm each over 31.25 ms; half_free/sham_free hold the same patches at zero displacement.",
        full_move="Right component receives +0.6 m/s once after withdrawal. Free runs receive zero kick. CRESSim contact and MPM damping used, no extra velocity decay.",
        side_tracker="When the straight finite cutting edge reaches particle height, record its side. Exclude opposite-side reference edges within the 2.5 mm rest graph. Geometric graph check, separate from the physical opening test.",
    ),
    half=dict(
        material_components_at_depth=int(load("refined", "half_load")["components"]),
        cases=half,
        pair_spacing_definition="Median signed axial center spacing of original neighbors straddling the cut, grouped by original height above 74.56 mm or below 58.56 mm. This includes rest particle spacing (~2 mm).",
    ),
    full=dict(
        native_pair_disagreeing_ids=int(np.sum(native["side"] != free["side"])),
        split_pair_disagreeing_ids=int(np.sum(right["side"] != right_free["side"])),
        finer_pair_disagreeing_ids=int(
            np.sum(
                load("refined4", "native_load")["side"]
                != load("refined4", "native_free")["side"]
            )
        ),
        particles_by_side=[int(left["keep"].sum()), int(right["keep"].sum())],
        components=2,
        native_extra_mm=dict(
            loaded=response(native, free, 1), passive=response(native, free, -1)
        ),
        independent_grids_extra_mm=dict(
            loaded=response(right, right_free, 1), passive=0
        ),
        independent_passive_note="Both loaded and unloaded composite replays use the same independent passive grid trajectory. Its zero additional response is imposed by construction. There is no inter-piece collision coupling after this hand-off.",
        independent_passive_absolute_after_withdraw_mm=increment(left, left["keep"]),
        handoff="At full depth in primary 7.629 us run. Each retained group keeps all stress/deformation history by a matched complete replay followed by masking the complementary particles.",
        no_load_full_cut_cm_max_z=float(free["end"][:, 2].max() * 100),
        directly_loaded_motion_reference="Additional axial center-of-mass displacement over the post-withdrawal interval, subtracting the same interval in the no-kick control. Restrict each comparison to IDs assigned to the same side in both GPU replays.",
    ),
    geometry=dict(
        native_postwithdraw_min_tool_clearance_mm=min(
            float(t["min_tool_clearance_m"]) for t in (native, free)
        )
        * 1000,
        during_motion="Linear tests across recorded kinematic updates (16 or 32 etc. substeps depending on dt). Cross the mid-plane within the finite blade footprint, excluding its 1 mm rim. Not an exhaustive physics-substep CCD test.",
        unique_crossings=len(events),
        events=sum(sum(e) for e in events.values()),
        tracked_particle=71186,
        tracked_min_signed_y_mm=float(native["probe_min_signed_y_mm"]),
        example_pair=pair,
        rest_pair_mm=pair_dist(rest),
        through_depth_pair_mm=pair_dist(native["full_depth"]),
        withdrawn_pair_mm=pair_dist(native["withdrawn"]),
        end_pair_mm=pair_dist(native["end"]),
    ),
    native_no_kick_sensitivity_by_dt_us=sensitivity,
    finer_native_extra_mm=dict(
        loaded=response(
            load("refined4", "native_load"), load("refined4", "native_free"), 1
        ),
        passive=response(
            load("refined4", "native_load"), load("refined4", "native_free"), -1
        ),
    ),
    notes=[
        "Single reference specimen. Native slicer, flat blade and contact envelope replace the previous wedge.",
        "Sensitivity includes a coarse immediate-handoff run which reached the grid-boundary guard after withdrawal; excluded from quantitative comparison. Independent-grid primary branch ran at the finer step.",
        "A separate repeat of the 7.629 us native full-cut run reached the grid boundary. Reported numeric results are from completed bounded runs. The native USD uses five full-particle keyframes from the measured run.",
        "USD and video are replays of CRESSim coordinates. USD nodes are selectable, the replay has no live PhysX interaction.",
    ],
)
(out / "results.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
)
print(
    "half pull",
    half["half_load"]["extra_vs_hold_mm"],
    "sham",
    half["sham_load"]["extra_vs_hold_mm"],
)
print(
    "native",
    report["full"]["native_extra_mm"],
    "split",
    report["full"]["independent_grids_extra_mm"],
    "passive absolute",
    report["full"]["independent_passive_absolute_after_withdraw_mm"],
)
print("fine", report["finer_native_extra_mm"])
print("geometry", report["geometry"])
