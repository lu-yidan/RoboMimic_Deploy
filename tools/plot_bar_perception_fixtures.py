"""Plot saved visible proposals over their source images, with bound inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from perception.orange_bar import CameraCalibration, project


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    fixtures = Path(__file__).resolve().parents[1] / 'tests/fixtures/bar_perception'
    cases = [('clean_initial', 'Clean frame'), ('high_bar_plus80', 'Raised upper rod'),
             ('orange_planar_distractor', 'Orange planar distractor'),
             ('partial_bar_occlusion', 'Partial occlusion'),
             ('heavy_occlusion', 'Heavy occlusion: abstain')]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), facecolor='white')
    manifest = {}
    for ax, (name, title) in zip(axes.flat, cases):
        folder = fixtures / name
        prediction = json.loads((folder / 'expected_prediction.json').read_text())
        calibration = CameraCalibration(**json.loads((folder / 'calibration.json').read_text()))
        with np.load(folder / 'measurement.npz', allow_pickle=False) as frame:
            ax.imshow(frame['rgb'])
            for target in prediction['targets']:
                ends, _ = project(target['visible_endpoints_world_m'], calibration,
                                  frame['camera_position'], frame['camera_rotation'])
                point, _ = project([target['target_world_m']], calibration,
                                   frame['camera_position'], frame['camera_rotation'])
                ax.plot(ends[:, 0], ends[:, 1], color='#00e5ff', linewidth=2)
                ax.scatter(point[:, 0], point[:, 1], marker='x', s=85, color='yellow', linewidth=2)
        ax.set_title(f"{title}\n{len(prediction['targets'])} visible proposals", fontsize=12)
        ax.axis('off')
        manifest[name] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in folder.iterdir() if p.is_file()}
    axes.flat[-1].axis('off')
    axes.flat[-1].text(.04, .85,
        'Saved-frame regression examples\n\nCyan: measured visible interval\nYellow: interval midpoint proposal\n\nKnown orange color and 15 mm radius\nCalibrated depth + supplied camera pose\n\nNo target identity or load permission\nNo closed-loop or real-camera claim',
        transform=axes.flat[-1].transAxes, va='top', fontsize=12, linespacing=1.7)
    fig.suptitle('Image-only rod geometry: portable frozen baseline', fontsize=18)
    fig.subplots_adjust(left=.015, right=.985, top=.90, bottom=.03,
                        wspace=.04, hspace=.23)
    for extension in ('png', 'pdf'):
        fig.savefig(args.output / f'visible_target_examples.{extension}', dpi=160)
    plt.close(fig)
    manifest['plot_source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
