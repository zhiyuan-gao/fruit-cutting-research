"""Side-view replay and tracked close-up, rendered from exported GPU particle positions."""

from pathlib import Path
import json
import subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont

root = Path(__file__).resolve().parents[2]
cache = root / ".cache/cressim-run/usd-frames"
out = root / "outputs/cressim-cucumber"
rest = np.load(root / "examples/cucumber/2026_0000/mpm.npz")["x"]
plane = (rest[:, 1].min() + rest[:, 1].max()) / 2
W, H = 1280, 672
fontdir = "/usr/share/fonts/truetype/dejavu/"


def font(size, bold=False):
    return ImageFont.truetype(
        fontdir + ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"), size
    )


title, sub, body, small = font(33, True), font(21, True), font(18), font(15)
navy = "#1b3434"
green = "#439575"
orange = "#d18147"
blue = "#3c68b0"


def patch(d, x, y, color, text):
    d.ellipse((x, y + 5, x + 11, y + 16), fill=color)
    d.text((x + 19, y), text, font=small, fill=navy)


def draw_frame(mode, index, rows, frames, sides, mask, half_grip, original_ids):
    full = mode != "half"
    row = rows[index]
    xyz = frames[index, mask]
    restsel = rest[original_ids[mask]]
    side = sides[mask]
    phase = row["phase"]
    split = full and index >= max(
        i for i, r in enumerate(rows) if r["phase"] == "through_depth"
    )
    load = phase in ("move_away", "opening")
    im = Image.new("RGB", (W, H), "#f5f5f0")
    d = ImageDraw.Draw(im)
    d.text(
        (36, 20),
        {
            "half": "Half cut and withdrawal",
            "native": "Full cut on the native grid",
            "separate": "Full cut with independent grids",
        }[mode],
        font=title,
        fill=navy,
    )
    d.text(
        (38, 68),
        "CRESSim-MPM   /   reference cucumber   /   114,517 simulated particles",
        font=body,
        fill=navy,
    )
    names = {
        "intact": "Before contact",
        "half_depth": "Downstroke",
        "through_depth": "Downstroke",
        "withdraw": "Withdrawal",
        "opening": "Opening test",
        "move_away": "One-sided motion test",
    }
    d.text((W - 38, 27), names[phase], anchor="ra", font=sub, fill=navy)
    d.text(
        (W - 38, 66),
        f"{row['time_s'] * 1000:.1f} ms",
        anchor="ra",
        font=body,
        fill=navy,
    )
    points = np.c_[(xyz[:, 1] - plane) * 1000, xyz[:, 2] * 1000]
    cutline = abs(restsel[:, 1] - plane) < 0.012
    center = np.median(points[cutline], axis=0)
    for bb, span, center_, label in [
        ((34, 113, 673, 544), 450, np.array([0, 122]), "Fixed view"),
        ((691, 113, 1246, 544), 105, center, "Cut region"),
    ]:
        left, t, r, b = bb
        d.rectangle(bb, fill="#ffffff", outline="#cdd8d2")
        panel = Image.new("RGB", (r - left - 2, b - t - 2), "#ffffff")
        pd = ImageDraw.Draw(panel)
        scale = (r - left) / span

        def project(y, z):
            return (
                (r - left) / 2 + (y - center_[0]) * scale,
                (b - t) / 2 - (z - center_[1]) * scale,
            )

        for y0, y1 in [((-0.18 - plane) * 1000, -12), (12, (0.18 - plane) * 1000)]:
            pd.rectangle([project(y0, 31), project(y1, 5)], fill="#d6d8cc")
        tip = row["tip_z_m"] * 1000
        pd.rectangle([project(-2, tip + 61.4), project(2, tip)], fill="#96a1ae")
        u, v = project(points[:, 0], points[:, 1])
        cols = np.where(side > 0, green, orange) if split else np.full(len(side), green)
        if not full and load:
            cols[half_grip[mask]] = blue
        for c in (green, orange, blue):
            select = (
                (cols == c) & (u > -2) & (u < r - left + 2) & (v > -2) & (v < b - t + 2)
            )
            for px, py in zip(u[select], v[select]):
                rad = max(1, scale * 0.44)
                pd.ellipse((px - rad, py - rad, px + rad, py + rad), fill=c)
        length = 50 if label == "Fixed view" else 10
        pd.line((18, b - t - 23, 18 + scale * length, b - t - 23), fill=navy, width=3)
        pd.text(
            (19, b - t - 45),
            f"{length} mm",
            fill=navy,
            font=small,
            stroke_width=2,
            stroke_fill="white",
        )
        pd.text(
            (14, 11), label, fill=navy, font=small, stroke_width=2, stroke_fill="white"
        )
        im.paste(panel, (left + 1, t + 1))
    patch(d, 40, 567, green, "Cucumber" if not split else "Loaded half")
    if split:
        patch(d, 210, 567, orange, "Passive half")
    if not full and load:
        patch(d, 210, 567, blue, "Pull patches")
    d.text(
        (1244, 565),
        "Longitudinal material slice, side view",
        anchor="ra",
        font=body,
        fill=navy,
    )
    if full:
        msg = {
            "intact": "The cucumber begins as one material body.",
            "half_depth": "The blade advances through the original specimen.",
            "through_depth": "At full depth, the material tracker identifies two connected components.",
            "withdraw": "The two components now evolve on separate MPM grids.",
            "move_away": "The green half receives an outward velocity of 0.6 m/s. The orange half receives no kick.",
        }
    else:
        msg = {
            "intact": "The cucumber begins as one material body.",
            "half_depth": "The blade stops inside the cucumber.",
            "withdraw": "The blade lifts out while the lower tissue remains joined.",
            "opening": "Two upper patches are moved apart by 4 mm each over 31.25 ms.",
        }
    if mode == "native":
        msg["through_depth"] = (
            "The two diagnostic colors track the materials on either side of the blade."
        )
        msg["withdraw"] = "The native CRESSim solver keeps both groups on one grid."
        msg["move_away"] = (
            "Only the green half receives the velocity kick. Both groups still share the native grid."
        )
    d.text((38, 617), msg[phase], fill=navy, font=body)
    return im


def export():
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
        "24",
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-crf",
        "19",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(out / "cressim-cucumber.mp4"),
    ]
    ff = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for mode in ("half", "native", "separate"):
        main_file = {
            "half": cache / "half_load.npz",
            "native": cache.parent / "refined/native_load.npz",
            "separate": cache / "split_load.npz",
        }[mode]
        main = np.load(main_file)
        rows = json.loads(str(main["rows"]))
        frames = main["frames"]
        sides = main["side"]
        if mode == "separate":
            other = np.load(cache / "split_passive.npz")
            assert np.array_equal(other["side"], sides)
            last = max(i for i, r in enumerate(rows) if r["phase"] == "through_depth")
            passive = other["keep"]
            of = other["frames"]
            frames[last + 1 :, passive] = of[last + 1 :, passive]
        original_ids = main["visual_ids"]
        mask = abs(rest[original_ids, 0] + 0.028) < 0.0023
        patchmask = (
            (abs(rest[:, 1] - plane) > 0.012)
            & (abs(rest[:, 1] - plane) < 0.035)
            & (rest[:, 2] > 0.07856)
        )
        for i in range(len(rows)):
            im = draw_frame(
                mode,
                i,
                rows,
                frames,
                sides[original_ids],
                mask,
                patchmask[original_ids],
                original_ids,
            )
            hold = 5 + (
                15
                if i == 0
                or i == len(rows) - 1
                or (i + 1 < len(rows) and rows[i]["phase"] != rows[i + 1]["phase"])
                else 0
            )
            frame = im.tobytes()
            for _ in range(hold):
                ff.stdin.write(frame)
    ff.stdin.close()
    if ff.wait():
        raise RuntimeError("Encoding failed")


if __name__ == "__main__":
    export()
