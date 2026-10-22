"""Audit declared initial geometry independently of later control outcomes."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mujoco
import numpy as np
from training.wall_support_env import WallSupportEnv
from training.climb_robustness import apply_rung_offsets


def inspect_initial(env, qpos):
    # Scratch data: never change a running episode or infer static equilibrium.
    m = env.model
    d = mujoco.MjData(m)
    d.qpos[:] = qpos
    mujoco.mj_forward(m, d)
    expected = {m.body(name).id: rung for name, rung in [
        ('left_ankle_roll_link', 'rung_00'), ('right_ankle_roll_link', 'rung_00'),
        ('hook', 'rung_04'), ('right_hook', 'rung_04')]}
    contacts = []
    for c in d.contact:
        ba, bb = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
        ina, inb = ba in env.robot_bodies, bb in env.robot_bodies
        if not (ina or inb):
            continue
        external = None if ina == inb else m.geom(c.geom2 if ina else c.geom1).name
        intended = ina != inb and expected.get(ba if ina else bb) == external
        contacts.append(dict(bodies=[m.body(ba).name, m.body(bb).name],
                             external_geom=external, distance_m=float(c.dist), intended=intended))
    bad = [c for c in contacts if not c['intended'] and c['distance_m'] < -1e-5]
    intended_depth = max([0.] + [-c['distance_m'] for c in contacts if c['intended']])
    return dict(passed=not bad and intended_depth <= .0007,
                unexpected_overlaps=bad, intended_penetration_peak_m=intended_depth,
                contacts=contacts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--suite', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    protocol = json.loads((a.suite/'protocol.json').read_text())
    path = Path(protocol['reference'])
    with np.load(path) as plan:
        q = plan['qpos'][0].copy()
    results = []
    for name, offsets in protocol['cases']:
        env = WallSupportEnv(ROOT/'outputs/wall_support_bundle_pd15_v2')
        try:
            env.reset()
            variant = apply_rung_offsets(env, offsets)
            results.append(dict(name=name, offsets=offsets, geometry_sha256=variant['geometry_sha256'],
                                **inspect_initial(env, q)))
        finally:
            env.close()
    result = dict(scope='Initial geometry only; does not establish force equilibrium or dynamic success. '
                        'Unexpected overlap tolerance 10 um, intended penetration 0.7 mm, as dense reference audit.',
                  plan=str(path.resolve()), plan_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  results=results)
    with a.out.open('x') as f:
        json.dump(result, f, indent=2)
    print(json.dumps([dict(name=r['name'], passed=r['passed'], overlaps=r['unexpected_overlaps']) for r in results], indent=2))


if __name__ == '__main__':
    main()
