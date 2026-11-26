#!/usr/bin/env bash
# Run in the prepared fruit-cutting-research workspace, with a reference cucumber and tool packs.
set -euo pipefail
cd "$(dirname "$0")/../.."
runner="${COUPON_PYTHON:-./python.sh}"
out=outputs/cut-physics
trial() {
    local label="$1" mode="$2"
    shift 2
    "$runner" cut_physics_trial.py "$mode" --out "$out/$label" "$@"
}
trial air air
trial indent indent --contact-samples
trial mirror indent --wrist-x-mm -100
trial half half --timeline
trial full80 full --timeline
trial repeat full
for k in 40 160; do trial "full$k" full --keff "$k"; done
for k in 40 80 160; do
    trial "fast$k" full --speed .10 --keff "$k"
    trial "offset$k" full --lateral-mm .5 --keff "$k"
done
trial fine_dt full --dt .0000025
trial indent-halfdt indent --dt .0000025
for mode in full indent; do trial "fixed-k-$mode" "$mode" --dt .0000025 --fixed-contact-stiffness; done
trial fine_space indent --spacing .0005 --dt .0000025
trial refine-full full --spacing .0005 --dt .0000025
trial refine-offset full --spacing .0005 --dt .0000025 --lateral-mm .25
trial stagger full --stagger
trial stagger-offset full --stagger --lateral-mm .5
trial no_friction indent --friction 0
"$runner" scripts/cut_physics/piece_probe.py "$out/half" "$out/full80"
# These two checks use the existing Taichi and USD packages in the repository image.
./python.sh scripts/cut_physics/native_probe.py indent --out "$out/native-indent"
./python.sh scripts/cut_physics/native_probe.py full --out "$out/native-full"
./python.sh scripts/cut_physics/archive_probe.py
./python.sh scripts/cut_physics/export_stage.py "$out/full80" "$out/coupon.usdc"
"$runner" scripts/cut_physics/report.py
