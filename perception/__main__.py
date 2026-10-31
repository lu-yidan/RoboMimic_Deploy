"""Run visible-only rod proposals on a saved calibrated RGB-D frame."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from . import CameraCalibration, select_visible_grasp_targets


def predict(input_dir):
    source = Path(input_dir)
    calibration = CameraCalibration(**json.loads(
        (source / "calibration.json").read_text(encoding="utf-8")))
    with np.load(source / "measurement.npz", allow_pickle=False) as frame:
        return select_visible_grasp_targets(
            frame["rgb"], frame["depth"], calibration,
            frame["camera_position"], frame["camera_rotation"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = predict(args.input_dir)
    # Exclusive output creation preserves previous evidence.
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "prediction.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    manifest = {"inputs": {
        name: hashlib.sha256((args.input_dir / name).read_bytes()).hexdigest()
        for name in ("calibration.json", "measurement.npz")},
        "sources": {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                    for name in ("__main__.py", "orange_bar.py", "visible_targets.py")},
        "target_count": len(result["targets"]), "load_permission": False,
        "scope": "saved RGB-D geometry only; no policy or physics execution"}
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "targets": len(result["targets"])}))


if __name__ == "__main__":
    main()
