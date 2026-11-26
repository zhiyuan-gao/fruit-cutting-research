"""Render the Chinese comparison from CRESSim replays and their measured displacements."""

from pathlib import Path
from functools import lru_cache
import json
import subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont

root = Path(__file__).resolve().parents[2]
raw = root / ".cache/cressim-run"
out = root / "outputs/cressim-cucumber"
metrics = json.loads((out / "results.json").read_text())
rest = np.load(root / "examples/cucumber/2026_0000/mpm.npz")["x"]
plane = (rest[:, 1].min() + rest[:, 1].max()) * 0.5
sids = np.flatnonzero(abs(rest[:, 0] + 0.028) < 0.0023)
green = "#208469"
orange = "#cf6d3a"
blue = "#3d70ac"
ink = "#172d2b"
muted = "#526861"
bg = "#f4f5ef"
white = "#ffffff"
line = "#ccd7ce"
W, H = 1440, 810
fps = 24


@lru_cache(None)
def font(size, bold=False):
    weight = "Bold" if bold else "Regular"
    return ImageFont.truetype(
        f"/usr/share/fonts/opentype/noto/NotoSansCJK-{weight}.ttc", size, index=2
    )


def label(d, x, y, txt, size=24, fill=ink, bold=False, anchor="lt"):
    d.text((x, y), txt, font=font(size, bold), fill=fill, anchor=anchor)


class Track:
    def __init__(self, name):
        with np.load(raw / (name + ".npz")) as src:
            self.rows = json.loads(str(src["rows"]))
            ids = src["visual_ids"]
            pos = np.searchsorted(ids, sids)
            assert np.array_equal(ids[pos], sids)
            self.yz = src["frames"][:, pos, 1:3].copy() * 1000
            self.yz[:, :, 0] -= plane * 1000
            self.side = src["side"][sids]
            self.keep = src["keep"][sids]


def load_chapter(mode):
    if mode == "half":
        return Track("refined/half_load"), Track("refined/half_free")
    if mode == "native":
        return Track("refined/native_load"), Track("refined/native_free")
    push = Track("usd-frames/split_load")
    free = Track("refined-early/split_free")
    left = Track("usd-frames/split_passive")
    assert np.array_equal(push.keep, ~left.keep) and np.array_equal(
        free.keep, push.keep
    )
    split = max(i for i, r in enumerate(push.rows) if r["phase"] == "through_depth")
    for branch in [push, free]:
        branch.yz[split + 1 :, left.keep] = left.yz[split + 1 :, left.keep]
    return push, free


def legend(d, x, y, col, text):
    d.ellipse((x, y + 5, x + 13, y + 18), fill=col)
    label(d, x + 22, y, text, 19)


def panel(track, index, mode, loaded):
    pw, ph = 665, 386
    im = Image.new("RGB", (pw, ph), white)
    d = ImageDraw.Draw(im)
    row = track.rows[index]
    half = mode == "half"
    scale = 2.52 if half else 1.39
    xcenter = 0 if half else -18
    base = ph - 16

    def project(y, z):
        return ((y - xcenter) * scale + pw / 2, base - z * scale)

    for y0, y1 in ((-180 - plane * 1000, -12), (12, 180 - plane * 1000)):
        d.rectangle([project(y0, 31), project(y1, 5)], fill="#cdd2c3")
    tip = row["tip_z_m"] * 1000
    d.rectangle([project(-2, tip + 61.4), project(2, tip)], fill="#738490")
    # The positions come from the same fixed material slice in all replays.
    y, z = track.yz[index].T
    u, v = project(y, z)
    depth_end = max(
        [i for i, r in enumerate(track.rows) if r["phase"] == "through_depth"],
        default=len(track.rows),
    )
    phase = row["phase"]
    masks = [(np.ones(len(y), bool), green)]
    if not half and index >= depth_end:
        masks = [(track.side < 0, orange), (track.side > 0, green)]
    if half and phase == "opening":
        axial = np.abs(rest[sids, 1] - plane)
        grip = (axial > 0.012) & (axial < 0.035) & (rest[sids, 2] > 0.07856)
        masks = [(~grip, green), (grip, blue)]
    # Stamp short horizontal/vertical strokes in pixel arrays for legible particles.
    pix = np.asarray(im).copy()
    for mask, color in masks:
        visible = mask & (u > 1) & (v > 1) & (u < pw - 2) & (v < ph - 2)
        px = np.rint(u[visible]).astype(int)
        py = np.rint(v[visible]).astype(int)
        rgb = tuple(bytes.fromhex(color[1:]))
        for ox, oy in [(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1)]:
            pix[py + oy, px + ox] = rgb
    im = Image.fromarray(pix)
    d = ImageDraw.Draw(im)
    bar = 20 if half else 50
    d.rounded_rectangle((10, 10, 115, 52), radius=7, fill="#f4f6f2")
    label(d, 18, 17, f"{bar} mm", 17, fill=muted)
    d.line((18, 44, 18 + bar * scale, 44), fill=muted, width=2)
    return im


titles = {
    "half": "切到一半，拔刀后轻轻拉开",
    "native": "完全切透，只推右块",
    "split": "切透后分成两个独立物体",
}
subs = {
    "half": "半切试验  /  同一截面放大",
    "native": "原生 CRESSim-MPM  /  两块共用一个网格",
    "split": "独立网格  /  两块分别计算运动",
}
phase_names = {
    "intact": "下刀前",
    "half_depth": "下刀",
    "through_depth": "继续下切",
    "withdraw": "撤刀",
    "opening": "向外轻拉",
    "move_away": "观察两块运动",
}


def draw(mode, idx, push, free, final):
    r = push.rows[idx]
    assert np.isclose(r["time_s"], free.rows[idx]["time_s"])
    half = mode == "half"
    im = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(im)
    label(d, 40, 24, subs[mode], 22, muted)
    label(d, 40, 62, titles[mode], 47, ink, True)
    label(d, 1400, 39, "CRESSim-MPM   /   黄瓜切割实验", 22, muted, anchor="rt")
    label(
        d,
        1400,
        84,
        f"{phase_names[r['phase']]}   ·   {r['time_s'] * 1000:.1f} ms",
        24,
        ink,
        True,
        "rt",
    )
    for j, branch in enumerate([push, free]):
        left = 40 + j * 695
        d.rounded_rectangle(
            (left, 140, left + 665, 585), radius=12, fill=white, outline=line, width=2
        )
        text = (
            ("两侧各拉 4 mm" if half else "只给右块向右的初速度")
            if j == 0
            else ("夹住两侧，保持不动" if half else "两块均不推，自然回弹")
        )
        label(
            d,
            left + 16,
            158,
            ("实验组   " if j == 0 else "对照组   ") + text,
            24,
            ink,
            True,
        )
        im.paste(panel(branch, idx, mode, j == 0), (left + 1, 193))
    if half:
        legend(d, 44, 602, blue, "夹持区域")
        legend(d, 225, 602, green, "黄瓜组织")
    else:
        legend(d, 44, 602, orange, "左块")
        legend(d, 170, 602, green, "右块")
    label(d, 1397, 602, "同一物理时刻  ·  固定侧视镜头", 20, muted, anchor="rt")
    d.rounded_rectangle((40, 649, 1400, 778), radius=13, fill="#e6ece3")
    if final:
        if half:
            label(
                d,
                61,
                668,
                f"拉开后，切口上方邻近粒子的间距额外增加 {metrics['half']['cases']['half_load']['extra_vs_hold_mm']['upper']:.2f} mm，底部几乎无变化。",
                29,
                ink,
                True,
            )
            label(
                d,
                61,
                720,
                "两组都经历相同的下刀和撤刀。数值为两组最终粒子间距的差。",
                22,
                muted,
            )
        else:
            data = metrics["full"][
                "native_extra_mm" if mode == "native" else "independent_grids_extra_mm"
            ]
            label(
                d, 61, 666, f"右块额外移动   {data['loaded']:.2f} mm", 32, green, True
            )
            label(
                d,
                715,
                666,
                f"左块额外移动   {data['passive']:.2f} mm",
                32,
                orange,
                True,
            )
            note = (
                "扣除无载回弹后，两块几乎一起向右移动。"
                if mode == "native"
                else "左块在两组里共用同一条独立轨迹，仍然会自然回弹。此试验只检验向外推开。"
            )
            label(d, 61, 722, note, 22, ink)
    else:
        if half:
            message = {
                "intact": "观察刀口上、下两处。",
                "half_depth": "刀停在黄瓜内部，底下还有组织相连。",
                "withdraw": "拔刀后，下一步只拉上方两处果肉。",
                "opening": "对比蓝色夹持区域之间的切口。两侧分别向外拉约 4 mm。",
            }[r["phase"]]
            detail = "切口上方和下方分别追踪原本相邻的粒子。"
        else:
            message = {
                "intact": "同一根黄瓜，同一条下刀轨迹。",
                "half_depth": "切刀向下进入黄瓜。",
                "through_depth": "刀继续向下，直到穿过底部。",
                "withdraw": "刀抬高到果肉上方，等待撤刀完成。",
                "move_away": "实验组右块获得 0.6 m/s 的向右初速度，再与旁边的不推对照比较。",
            }[r["phase"]]
            detail = (
                "切透后，橙色和绿色分别标出左、右两块。"
                if mode == "native"
                else "从切透时开始，左右两块各自在独立网格中运动。"
            )
        label(d, 61, 668, message, 28, ink, True)
        label(d, 61, 722, detail, 22, muted)
    return im


def summary():
    im = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(im)
    label(d, 65, 45, "CRESSim-MPM   /   黄瓜切割实验", 24, muted)
    label(d, 65, 107, "只推右块，左块多移动了多少？", 51, ink, True)
    label(d, 65, 203, "图中为扣除无载回弹后的左块位移", 25, muted)
    vals = [
        ("原生共用网格", metrics["full"]["native_extra_mm"]["passive"]),
        ("分为独立物体", metrics["full"]["independent_grids_extra_mm"]["passive"]),
    ]
    for j, (name, val) in enumerate(vals):
        y = 300 + j * 105
        label(d, 66, y, name, 30, ink, True)
        d.rounded_rectangle((340, y + 2, 1130, y + 43), radius=8, fill="#dce3da")
        if val:
            d.rounded_rectangle(
                (340, y + 2, 340 + 79 * val, y + 43), radius=8, fill=orange
            )
        else:
            d.line((340, y + 1, 340, y + 45), fill=orange, width=4)
        label(d, 1365, y, f"{val:.2f} mm", 34, orange, True, "rt")
    d.line((65, 534, 1374, 534), fill=line, width=2)
    label(d, 65, 577, "刀口附近仍有异常压缩", 31, ink, True)
    label(d, 65, 644, "一对果肉粒子从相距 2.04 mm 被压到约 0.01 mm。", 27, ink)
    label(
        d,
        65,
        704,
        "独立物体试验使用互不接触的网格。切块再次相撞时的表现仍需验证。",
        23,
        muted,
    )
    return im


def main():
    path = out / "cressim-cucumber-results-zh.mp4"
    stage = path.with_suffix(".part.mp4")
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{W}x{H}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(stage),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    framecount = 0

    def send(im, reps):
        nonlocal framecount
        data = im.tobytes()
        for _ in range(reps):
            proc.stdin.write(data)
        framecount += reps

    for mode in ["half", "native", "split"]:
        push, free = load_chapter(mode)
        print(mode, framecount / fps, flush=True)
        for i, r in enumerate(push.rows):
            last = i == len(push.rows) - 1
            phase_end = not last and push.rows[i + 1]["phase"] != r["phase"]
            reps = 4 if half_or_move(mode, r) else 2
            reps += 18 if i == 0 or phase_end else 0
            send(draw(mode, i, push, free, last), 80 if last else reps)
    print("summary", framecount / fps, flush=True)
    send(summary(), 100)
    proc.stdin.close()
    if proc.wait():
        raise RuntimeError("FFmpeg failed")
    stage.replace(path)
    print("end", framecount / fps, flush=True)


def half_or_move(mode, r):
    return mode == "half" or r["phase"] == "move_away"


if __name__ == "__main__":
    main()
