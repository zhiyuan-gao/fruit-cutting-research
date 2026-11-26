"""Audit recorded coupon runs and package the force–depth and separation comparisons."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/cut-physics"


def read(path):
    return json.loads(path.read_text())


def render(results, probes):
    font = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    chinese = font.exists()
    if chinese:
        font_manager.fontManager.addfont(font)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=font).get_name()

    def label(zh, en):
        return zh if chinese else en

    fig, axes = plt.subplots(2, 2, figsize=(11.9, 7.3), layout="constrained")
    green = "#067962"
    orange = "#b95625"
    blue = "#365a91"
    ax = axes[0, 0]
    for name, color, title in [
        ("full80", green, "0.05 m/s"),
        ("fast80", blue, "0.10 m/s"),
        (
            "offset80",
            orange,
            label("0.05 m/s，刀横移 0.5 mm", "0.05 m/s, 0.5 mm offset"),
        ),
    ]:
        csv = np.genfromtxt(OUT / name / "wrench.csv", delimiter=",", names=True)
        down = (
            csv["time_s"]
            <= results[name]["phase_end_steps"]["down"] * results[name]["dt_s"]
            + 1.0e-12
        )
        ax.plot(
            csv["depth_mm"][down], csv["Fz_N"][down], lw=1.7, label=title, color=color
        )
    ax.set(
        xlabel=label("刀刃低于试块顶面的距离（mm）", "Depth below coupon top (mm)"),
        ylabel=label("竖向反力（N）", "Axial reaction (N)"),
        title=label("同一试块的下刀反力", "Force during insertion"),
    )
    ax.legend(fontsize=8, frameon=False)
    ax = axes[0, 1]
    names = ["full80", "offset80", "stagger", "stagger-offset"]
    work = [results[n]["downstroke_work_mJ"] for n in names]
    bars = ax.bar(np.arange(4), work, color=[green, orange, green, orange], width=0.66)
    ax.bar_label(bars, fmt="%.2f")
    ax.set_xticks(
        np.arange(4),
        [
            label("规则采样\n原刀位", "Regular\ncenter"),
            label("规则采样\n横移", "Regular\noffset"),
            label("交错采样\n原刀位", "Staggered\ncenter"),
            label("交错采样\n横移", "Staggered\noffset"),
        ],
    )
    ax.set(
        ylabel=label("下刀做功（mJ）", "Insertion work (mJ)"),
        ylim=(0, 1.18 * max(work)),
        title=label(
            "相同材料参数，改变粒子排列或刀位",
            "Sensitivity to particle sampling and knife offset",
        ),
    )
    ax = axes[1, 0]
    xx = np.arange(2)
    for i, (side, c, t) in enumerate(
        [
            ("right", green, label("受推的一侧", "Loaded side")),
            ("left", orange, label("另一侧", "Other side")),
        ]
    ):
        values = [probes[n]["delta_out_mm"][side] for n in ["half", "full80"]]
        bars = ax.bar(xx + (i - 0.5) * 0.36, values, 0.33, color=c, label=t)
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=9)
    ax.set_xticks(xx, [label("半切撤刀", "Half cut"), label("全切撤刀", "Through cut")])
    ax.set(
        ylabel=label(
            "相对无载的额外位移（mm）", "Additional displacement vs unloaded (mm)"
        ),
        title=label("只向外推右侧 12 ms", "Push only the right side for 12 ms"),
        ylim=(0, 1.5),
    )
    ax.legend(frameon=False, fontsize=8)
    ax = axes[1, 1]
    k = [40, 80, 160]
    for prefix, c, t in [
        ("full", green, label("居中", "Centered")),
        ("offset", orange, label("横移 0.5 mm", "0.5 mm offset")),
    ]:
        ax.plot(
            k,
            [results[prefix + str(n)]["threshold_reached"] for n in k],
            "o-",
            lw=2,
            color=c,
            label=t,
        )
    ax.set(
        xticks=k,
        xlabel=r"$K_{\rm eff}$ (J/m²)",
        ylabel=label("达到阻力阈值的粒子数", "Particles reaching work threshold"),
        title=label("刀尖采样对阻力模型的影响", "Work threshold activation"),
    )
    ax.legend(frameon=False, fontsize=8)
    for a in axes.flat:
        a.spines[["top", "right"]].set_visible(False)
        a.grid(axis="y", alpha=0.17)
        a.set_axisbelow(True)
    fig.suptitle(
        label(
            "黄瓜果肉均匀试块 · 6 × 12 × 6 mm · 纯弹性仿真",
            "Homogeneous cucumber-flesh coupon · 6 × 12 × 6 mm · elastic simulation",
        ),
        size=17,
    )
    fig.savefig(OUT / "coupon-results.png", dpi=155)
    plt.close(fig)


def main():
    names = (
        "air indent mirror half full80 repeat full40 full160 fast40 fast80 fast160 offset40 offset80 "
        "offset160 fine_dt indent-halfdt fixed-k-full fixed-k-indent fine_space refine-full refine-offset "
        "stagger stagger-offset no_friction native-indent native-full"
    ).split()
    results = {n: read(OUT / n / "result.json") for n in names}
    probes = {n: read(OUT / n / "piece_probe.json") for n in ("half", "full80")}
    old = read(OUT / "archive-check.json")
    # Independently integrate the CSV windows, including the shorter boundary windows.
    max_error = 0.0
    for name in names:
        rows = np.genfromtxt(OUT / name / "wrench.csv", names=True, delimiter=",")
        durations = np.diff(np.r_[0.0, rows["time_s"]])
        with np.load(OUT / name / "substeps.npz") as trace:
            for cols, channel in [
                (("Fx_N", "Fy_N", "Fz_N"), "force_impulse"),
                (("Mx_Nm", "My_Nm", "Mz_Nm"), "moment_impulse"),
            ]:
                estimate = np.column_stack([rows[c] for c in cols]).T @ durations
                target = trace[channel].sum(0, dtype=float)
                error = float(np.max(abs(estimate - target)))
                max_error = max(max_error, error)
                np.testing.assert_allclose(estimate, target, rtol=1.0e-6, atol=1.0e-10)
    clear = probes["full80"]["already_open_push"]
    main_case = results["full80"]
    clean = [r for r in results.values() if "elastic_only" in r]
    with np.load(OUT / "air" / "substeps.npz") as air:
        zero_wrench = not np.any(air["force_impulse"]) and not np.any(
            air["moment_impulse"]
        )
    checks = dict(
        archive_positive_control=old["entries"] == [1, 0]
        and old["inside_at_end"] == [1, 0],
        no_load=zero_wrench and results["air"]["cycle_work_mJ"] == 0.0,
        eccentric_torque_sign=results["indent"]["My_at_Fz_peak_Nm"] > 0.0
        and results["mirror"]["My_at_Fz_peak_Nm"] < 0.0,
        contact_point_reintegration=results["indent"]["nonzero_point_impulses"] > 0,
        no_observed_coupon_center_entry=all(
            r["segment_entries"] == 0 and r["max_inside_particles"] == 0 for r in clean
        ),
        positive_volume=all(r["min_J"] > 0 for r in clean),
        partial_connected=probes["half"]["components"] == 1,
        through_disconnected=probes["full80"]["components"] == 2,
        clear_push_no_drag=max(clear["normal_contact_Ns"]) == 0.0
        and abs(clear["left_extra_mm"]) < 0.001
        and clear["right_extra_mm"] > 0.1,
        recontact_and_release=probes["full80"]["loaded"][1]["pair_normal_impulse_Ns"]
        > 0.0
        and probes["full80"]["loaded"][2]["end_pair_impulse_Ns"] == 0.0
        and probes["full80"]["loaded"][2]["upper_gap_mm"]
        > probes["full80"]["loaded"][1]["upper_gap_mm"],
        stable_ids=all(
            row["stable_side"] for p in probes.values() for row in p["loaded"]
        ),
    )
    with np.load(OUT / "full80" / "substeps.npz") as raw:
        total = float(raw["tool_work"].sum(dtype=float))
        final = float(raw["energy"][-1].sum())
        sliding = float(raw["friction_loss"].sum(dtype=float))
    report = dict(
        sample="Synthetic clamped elastic coupon, seed-2026 cucumber Pulp properties. No robot measurements.",
        physical_setup="Straight vertical translation, actual knife mesh, no gravity, no board, fixture released for manipulation.",
        algorithms={
            "reference": "Single-cut TLMPM, quadratic reference stencils, Eq. 8 compatibility, PIC/FLIP with grid velocity advection and one gradient update. No MUSL re-scatter or static-friction history. Particle sphere closest-point contact, predicted gap, kinetic Coulomb friction. Work-limited forward impulse at the edge.",
            "native": "Pinned CulinaryCut elastic MLS-MPM control. Matched coupon and trajectory, Eulerian clamp nodes, default SDF contact, no separate damage or cut topology, no tip assistance or bulk damping.",
        },
        cases=results,
        manipulation=probes,
        archive_control=old,
        checks=checks,
        max_abs_window_integration_error_Ns_or_Nms=max_error,
        center_case_energy_mJ={
            "tool_work": 1000 * total,
            "final_kinetic_plus_strain": 1000 * final,
            "sliding_work_estimate": 1000 * sliding,
            "balance_after_sliding_estimate": 1000 * (total - final - sliding),
            "directional_gate_work_diagnostic": main_case["cumulative_gate_work_mJ"],
        },
        interpretation="Directional work is not debited a second time. Sliding work uses pre-impulse relative velocity; remaining work includes numerical dissipation and contact errors. Position and sampling sensitivity preclude identifying K_eff from this coupon.",
    )
    (ROOT / "records/cutting-physics-coupon.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    render(results, probes)
    print(json.dumps(checks, indent=2))
    if not all(checks.values()):
        raise SystemExit("Coupon diagnostic failed; inspect the individual checks")


if __name__ == "__main__":
    main()
