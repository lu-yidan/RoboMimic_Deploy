"""CPU-only pending motion selection, independent of the active tracker."""
from pathlib import Path
import glob


class MotionCatalog:
    def __init__(self, initial, patterns, root):
        initial = Path(initial).resolve()
        paths = [initial]
        for pattern in patterns:
            pattern = Path(str(pattern)).expanduser()
            if not pattern.is_absolute():
                pattern = Path(root) / pattern
            paths.extend(Path(p).resolve() for p in sorted(glob.glob(str(pattern))))
        self.paths = list(dict.fromkeys(paths))
        self.index = 0

    @property
    def selected(self):
        return self.paths[self.index]

    def move(self, delta):
        self.index = (self.index + delta) % len(self.paths)
        return self.selected

    def describe(self):
        return f'[Motion {self.index+1}/{len(self.paths)}] {self.selected.name} | Left/Right: select; R1: play from current state'


def play_selected(catalog, model, data, fsm, state, output):
    """Load transactionally and queue playback without changing simulated state."""
    from common.utils import FSMCommand, FSMStateName
    if fsm.cur_policy.name not in (FSMStateName.FIXEDPOSE, FSMStateName.LOCOMODE,
                                    FSMStateName.SKILL_AMP, FSMStateName.SKILL_SCALEBFM):
        raise ValueError("Start from FixedPose, Loco or AMP")
    tracker = fsm.scalebfm_policy
    if catalog.selected != tracker.motion_path:
        tracker.load_motion(catalog.selected)
    state.skill_cmd = FSMCommand.CMD_SCALEBFM
