"""Side-by-side playback of measured half- and through-cut cross sections (requires ffmpeg)."""

from pathlib import Path
import json
import subprocess
import sys

import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/mpm_cutting"))
from pieces import SingleCutPieces  # noqa: E402
from reference_cut import knife_geometry  # noqa: E402

OUT = ROOT / "outputs/cut-physics"
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"


def prepare(name):
    folder = OUT / name
    d = json.loads((folder / "result.json").read_text())
    loading = np.load(folder / "timeline.npz")
    released = np.load(folder / "manipulation.npz")
    x = np.concatenate([loading["x"], released["x"]])
    tracker = SingleCutPieces(x[0], d["spacing_m"])
    tags = np.concatenate([loading["side"], released["side"]])
    ids = np.stack([tracker.update(np.choose(s, [0, -1, 1])).copy() for s in tags])
    time = np.r_[loading["time_s"], loading["time_s"][-1] + released["time_s"]]
    z = np.r_[loading["knife_z"], released["knife_z"]]
    t1 = d["dt_s"] * d["phase_end_steps"]["down"]
    t2 = loading["time_s"][-1]
    mask = np.isclose(x[0, :, 0], 0.0005)
    return (
        x[:, mask, :],
        ids[:, mask],
        z,
        time,
        [0, t1, t2, t2 + 0.012, t2 + 0.046, t2 + 0.091],
    )


def main():
    small, title, large = [ImageFont.truetype(FONT, s) for s in (22, 28, 36)]
    cases = [prepare(n) for n in ["half", "full80"]]
    verts, faces, _ = knife_geometry(ROOT)
    section = trimesh.Trimesh(verts, faces, process=False).section(
        plane_normal=[1, 0, 0], plane_origin=[0.0005, 0, 0]
    )
    outlines = [path[:, 1:] * 1000 for path in section.discrete]
    labels = [
        "下刀 / Insertion",
        "撤刀 / Withdrawal",
        "向外推右侧 / Push right",
        "反向压合 / Press together",
        "再次分开 / Reopen",
    ]
    frames = [66, 42, 54, 54, 66]
    canvas = (1280, 640)
    proc = subprocess.Popen(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            "1280x640",
            "-r",
            "20",
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-crf",
            "19",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(OUT / "coupon-cuts.mp4"),
        ],
        stdin=subprocess.PIPE,
    )
    for phase, n in enumerate(frames):
        for j in range(n):
            u = j / (n - 1)
            image = Image.new("RGB", canvas, "#f7f7f3")
            draw = ImageDraw.Draw(image)
            draw.text(
                (34, 16), "黄瓜果肉试块  /  一刀切到哪里？", font=large, fill="#20362e"
            )
            draw.text((36, 76), labels[phase], font=title, fill="#315f50")
            for k, (positions, ids, z, time, stops) in enumerate(cases):
                left = 28 + 622 * k
                draw.text(
                    (left + 14, 128),
                    ["半切撤刀  /  Half cut", "全切撤刀  /  Through cut"][k],
                    font=title,
                    fill="#23352f",
                )
                panel = Image.new("RGB", (598, 335), "#e9efeb")
                g = ImageDraw.Draw(panel)
                t = (1 - u) * stops[phase] + u * stops[phase + 1]
                index = int(np.argmin(abs(time - t)))
                scale = (
                    29.0 / (1 + 1.42 * max(0.0, u - 0.28) / 0.72)
                    if phase == 4
                    else 29.0
                )

                def screen(y, h):
                    return 297 + y * scale, 265 - h * scale

                for path in outlines:
                    g.polygon(
                        [screen(y, h + z[index] * 1000) for y, h in path],
                        fill="#7b8790",
                    )
                width = 0.44 * scale
                for point, component in zip(positions[index], ids[index]):
                    a, b = screen(point[1] * 1000, point[2] * 1000)
                    g.ellipse(
                        (a - width, b - width, a + width, b + width),
                        fill=["#548760", "#33765f", "#a0a84f"][component],
                        outline="#385d47",
                    )
                g.line((20, 307, 20 + 2 * scale, 307), fill="#23352f", width=3)
                g.text((20, 274), "2 mm", font=small, fill="#23352f")
                image.paste(panel, (left, 180))
                count = len(np.unique(ids[index]))
                draw.text(
                    (left + 9, 524),
                    f"物理时间 {1000 * time[index]:.0f} ms   ·   连通块 {count}",
                    font=small,
                    fill="#2e4439",
                )
            draw.text(
                (36, 589),
                "同一把刀、同一截面。两侧按动作阶段对齐。圆点是实际仿真粒子。",
                font=small,
                fill="#4b5f55",
            )
            proc.stdin.write(image.tobytes())
    proc.stdin.close()
    if proc.wait():
        raise RuntimeError("ffmpeg encoding failed")


if __name__ == "__main__":
    main()
