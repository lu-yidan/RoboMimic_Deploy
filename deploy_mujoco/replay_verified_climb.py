"""Interactive kinematic replay of audited physics recordings; never runs a policy."""
import argparse
import json
from pathlib import Path
import time
import mujoco
import numpy as np
from training.wall_support_env import WallSupportEnv
from training.repeated_rung_errors import apply_rung_offsets
from tools.audit_multirung_cycle import sha


def load_recording(path):
    manifest = json.loads((path/'manifest.json').read_text())
    if any(manifest.get(key) for key in ('contact_admission','timed_loading','scene_compiler')):
        raise ValueError('This baseline replay entry does not support experimental controllers or compiled scene variants')
    summary = json.loads((path/'summary.json').read_text())
    audit = json.loads((path/'independent_audit.json').read_text())
    if not summary['passed'] or not audit['passed']:
        raise ValueError('Successful runtime and independent audit required for this demo entry point')
    assert sha(path/'trajectory.npz') == summary['trajectory_sha256'] == audit['trajectory_sha256']
    assert sha(path/'reference.npz') == manifest['plan_sha256']
    assert all(sha(path/'source_snapshot'/name) == digest for name, digest in manifest['sources'].items())
    if manifest.get('acceptance') == 'physical-v2':
        final = json.loads((path/'v2_multicycle_audit.json').read_text())
        assert final['passed'] and final['physical_passed'] and final['policy_passed']
        for key, name in [('manifest_sha256', 'manifest.json'), ('summary_sha256', 'summary.json'),
                          ('trajectory_sha256', 'trajectory.npz'), ('independent_audit_sha256', 'independent_audit.json'),
                          ('policy_audit_sha256', 'policy_transfer_audit.json')]:
            assert final[key] == sha(path/name)
        bundle_hash = manifest['bundle_manifest_sha256']
    else:
        bundle_hash = audit['bundle_manifest_sha256']
    if 'geometry_observation_roles' in manifest:
        policy = json.loads((path/'policy_transfer_audit.json').read_text())
        assert policy['passed'] and policy['manifest_sha256'] == sha(path/'manifest.json')
        assert policy['trajectory_sha256'] == summary['trajectory_sha256']
    else:
        rendered = json.loads((path/'render_manifest.json').read_text())
        assert rendered['provenance_sha256'] == sha(path/'manifest.json')
        assert rendered['summary_sha256'] == sha(path/'summary.json')
    env = WallSupportEnv(manifest['bundle'])
    try:
        assert sha(env.bundle/'manifest.json') == bundle_hash
        env.reset()
        variant = json.loads((path/'scene_variant.json').read_text())
        assert apply_rung_offsets(env, variant['offsets']) == variant
        with np.load(path/'trajectory.npz') as z:
            recording = {key: z[key].copy() for key in ('qpos', 'qvel', 'time_s')}
        with np.load(path/'reference.npz') as z:
            ends = z['cycle_ends_s'].copy()
        assert all(np.isfinite(array).all() for array in recording.values())
        assert len(recording['qpos']) == len(recording['qvel']) == len(recording['time_s'])
        assert np.allclose(np.diff(recording['time_s']), .02, atol=1e-8, rtol=0)
        assert abs(recording['time_s'][-1]-ends[-1]) < 1e-6
        return env, recording, ends
    except Exception:
        env.close()
        raise


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--speed', type=float, default=4.)
    p.add_argument('--loop', action='store_true')
    p.add_argument('--verify-only', action='store_true')
    a = p.parse_args()
    if not np.isfinite(a.speed) or not .1 <= a.speed <= 20.:
        raise ValueError('Replay speed must be between 0.1 and 20')
    env, data, ends = load_recording(a.run)
    try:
        report = dict(verified=True, cycles=len(ends), horizon_s=float(data['time_s'][-1]),
            rise_m=float(data['qpos'][-1, 2]-data['qpos'][0, 2]), run=str(a.run.resolve()),
            mode='Recorded kinematic replay only; no policy inference, physics stepping or robot connection.')
        print(json.dumps(report), flush=True)
        if a.verify_only:
            return
        import mujoco.viewer
        state = dict(paused=False, restart=False)
        def key_callback(key):
            if key == 32:
                state['paused'] = not state['paused']
            elif key in (ord('R'), ord('r')):
                state['restart'] = True
        for geom in range(env.model.ngeom):
            if (env.model.geom(geom).name or '').startswith('rung_'):
                env.model.geom_rgba[geom] = [.55, .63, .68, 1.]
        with mujoco.viewer.launch_passive(env.model, env.data, key_callback=key_callback) as viewer:
            viewer.opt.geomgroup[3], viewer.opt.geomgroup[4] = 0, 1
            viewer.cam.azimuth, viewer.cam.elevation, viewer.cam.distance = 65., -8., 2.5
            cursor, previous = 0., time.perf_counter()
            while viewer.is_running():
                now = time.perf_counter()
                if state['restart']:
                    cursor, state['restart'], state['paused'] = 0., False, False
                elif not state['paused']:
                    cursor += (now-previous)*a.speed
                previous = now
                if cursor >= data['time_s'][-1]:
                    if a.loop:
                        cursor %= float(data['time_s'][-1])
                    else:
                        cursor, state['paused'] = float(data['time_s'][-1]), True
                index = min(int(np.searchsorted(data['time_s'], cursor)), len(data['time_s'])-1)
                with viewer.lock():
                    env.data.qpos[:] = data['qpos'][index]
                    env.data.qvel[:] = data['qvel'][index]
                    env.data.time = float(data['time_s'][index])
                    mujoco.mj_kinematics(env.model, env.data)
                    viewer.cam.lookat[:] = env.data.xpos[env.pelvis]+np.array([0., 0., .05])
                cycle = min(int(np.searchsorted(ends, cursor, side='right'))+1, len(ends))
                rise = float(data['qpos'][index, 2]-data['qpos'][0, 2])
                viewer.set_texts((None, None,
                    'RECORDED REPLAY - not live simulation\nTime / cycle / root rise\nControls',
                    f'{a.speed:g}x | {"PAUSED" if state["paused"] else "PLAYING"}\n'
                    f'{cursor:.2f} s / {cycle} of {len(ends)} / {rise:.3f} m\nSpace: pause | R: restart | Mouse: camera'))
                viewer.sync()
                time.sleep(max(0., 1./60.-(time.perf_counter()-now)))
    finally:
        env.close()


if __name__ == '__main__':
    main()
