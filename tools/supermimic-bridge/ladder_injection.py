"""Inject the climbing ladder into a mjlab scene via SceneCfg.spec_fn.

Root cause (2026-10-31): SuperMimic's tracking env builds `SceneCfg(terrain=plane)` -- flat ground only.
Our reference has the robot hanging from ladder rungs, so with no ladder the robot finds no support and
the policy cannot survive the imagination horizon (surv = 0.000) no matter how the clip is prepared.

Geometry copied from our MuJoCo scene (extracted from the baseline model):
  rungs    capsule, radius 0.015 m, half-length 0.72 m, axis +y, at x ~= 0.276-0.320, z = 0.3 + 0.25*k
  rails    box, half-size (0.012, 0.035, 1.907), at x ~= 0.756, y = +-0.7, z ~= 1.9

Usage: patch the tracking env config's SceneCfg with spec_fn=add_ladder.
"""
import mujoco

# (x, z) per rung; x varies because the calibration/geometry perturbations shift each rung slightly.
RUNGS = (
    (0.2762, 0.30), (0.2981, 0.55), (0.3200, 0.80), (0.3418, 1.05),
    (0.3637, 1.30), (0.3855, 1.55), (0.4074, 1.80), (0.4292, 2.05),
)
RUNG_RADIUS = 0.015
RUNG_HALF_LENGTH = 0.72
RAIL_X = 0.7563
RAIL_HALF = (0.012, 0.035, 1.9073)
RAIL_Z = 1.9009


def add_ladder(spec: mujoco.MjSpec) -> None:
    """Attach rungs and rails to the scene's worldbody as static geoms."""
    world = spec.worldbody

    for index, (x, z) in enumerate(RUNGS):
        geom = world.add_geom()
        geom.name = 'ladder_rung_%02d' % index
        geom.type = mujoco.mjtGeom.mjGEOM_CAPSULE
        geom.size = [RUNG_RADIUS, RUNG_HALF_LENGTH, 0.0]
        geom.pos = [x, 0.0, z]
        geom.quat = [0.70710678, 0.0, 0.70710678, 0.0]   # rotate the capsule axis from +z onto +y
        geom.contype = 1
        geom.conaffinity = 1

    for side, y in (('left', 0.7), ('right', -0.7)):
        geom = world.add_geom()
        geom.name = 'ladder_rail_' + side
        geom.type = mujoco.mjtGeom.mjGEOM_BOX
        geom.size = list(RAIL_HALF)
        geom.pos = [RAIL_X, y, RAIL_Z]
        geom.contype = 1
        geom.conaffinity = 1


def probe() -> None:
    """Compile a bare scene with the ladder to prove the injection is valid."""
    spec = mujoco.MjSpec()
    add_ladder(spec)
    model = spec.compile()
    names = [model.geom(i).name for i in range(model.ngeom)]
    print('geoms:', len(names))
    print(names[:4], '...', names[-2:])


if __name__ == '__main__':
    probe()
