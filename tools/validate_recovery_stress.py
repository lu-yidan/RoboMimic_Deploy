"""Matched deployment-controller flat recovery and deterministic dynamics stress.
No DDS or hardware. Keeps host gains/target limiting unchanged under motor errors.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.validate_ft12k_sim import make_model, build_policy, run_case

SCENARIOS = {
    'nominal': (1., 1., 0.),
    'upper130': (1.3, 1., 0.),
    'motor080': (1., .8, 0.),
    'motor120': (1., 1.2, 0.),
    'delay10ms': (1., 1., 10.),
    'combined': (1.3, .8, 10.),
}
CACHE = {}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_policy(profile):
    if profile in CACHE:
        return CACHE[profile]
    if profile == 'amp':
        from unittest.mock import patch
        import onnxruntime as ort
        from common.ctrlcomp import StateAndCmd, PolicyOutput
        from tools.validate_amp_recovery import AmpAuditAdapter
        state, output = StateAndCmd(29), PolicyOutput(29)
        state.vel_cmd[:] = 0
        original = ort.SessionOptions
        def single_thread():
            options = original()
            options.intra_op_num_threads = options.inter_op_num_threads = 1
            return options
        with patch.object(ort, 'SessionOptions', single_thread):
            policy = AmpAuditAdapter(state, output)
    else:
        policy, state, output = build_policy(profile)
    CACHE[profile] = policy, state, output
    return CACHE[profile]


def worker(job):
    import mujoco
    profile, scenario, cases, folder = job
    folder = Path(folder)
    policy, state, output = load_policy(profile)
    model = make_model()
    upper, gain, delay = SCENARIOS[scenario]
    torso = model.body('torso_link').id
    upper_ids = []
    for body in range(1, model.nbody):
        parent = body
        while parent > 0 and parent != torso:
            parent = int(model.body_parentid[parent])
        if parent == torso:
            upper_ids.append(body)
    if upper != 1:
        model.body_mass[upper_ids] *= upper
        model.body_inertia[upper_ids] *= upper
        mujoco.mj_setConst(model, mujoco.MjData(model))
    results = []
    for case in cases:
        loads = []
        result, trace, obs = run_case(case, model, policy, state, output, record=True,
                                      motor_gain=gain, delay_ms=delay, load_trace=loads)
        assert result['initial_min_contact'] > -.005, (case['name'], result['initial_min_contact'])
        z, upright, hold = trace[:, model.nq + model.nv:][:, :3].T
        def persistent(mask):
            return bool(len(mask) >= 10 and np.any(np.convolve(mask.astype(int), np.ones(10, int), 'valid') >= 10))
        fallen = (z < .65) | (upright < .5)
        result.update(profile=profile, scenario=scenario, upper_mass_inertia_scale=upper,
                      success_10s=result['success_10s'] and result['unstable'] is None,
                      refall_after_upright=persistent(fallen & np.maximum.accumulate((z >= 1.15) & (upright >= .93))),
                      refall_after_hold1=persistent(fallen & np.maximum.accumulate(hold >= 1 - 1e-6)),
                      last10s_path_m=float(np.linalg.norm(np.diff(trace[-500:, :2], axis=0), axis=1).sum()))
        (folder / (case['name'] + '.json')).write_text(json.dumps(result, indent=2) + '\n')
        np.savez_compressed(folder / (case['name'] + '.npz'), state=trace, obs=obs,
                            load_500hz=np.asarray(loads), nq=model.nq, nv=model.nv,
                            qpos_initial=case['qpos'], upper_body_ids=upper_ids)
        results.append(result)
    return results


def aggregate(rows):
    result = dict(n=len(rows), success=sum(r['success_10s'] for r in rows),
                  reached_upright=sum(r['first_upright_s'] >= 0 for r in rows),
                  unstable=sum(r['unstable'] is not None for r in rows),
                  refall_after_upright=sum(r['refall_after_upright'] for r in rows),
                  refall_after_hold1=sum(r['refall_after_hold1'] for r in rows),
                  last10s_path_mean_m=float(np.mean([r['last10s_path_m'] for r in rows])))
    for field in ('peak_tau', 'peak_dq', 'peak_power', 'above90_cumulative_s', 'above90_longest_s'):
        values = np.array([r[field] for r in rows])
        result[field + '_p95'] = float(np.percentile(values.max(axis=1), 95))
        result[field + '_max'] = float(values.max())
        result[field + '_per_joint_p95'] = dict(zip(rows[0]['joint_names'], np.percentile(values, 95, axis=0).tolist()))
    result['per_direction'] = {d: dict(n=sum(r['direction'] == d for r in rows),
                                      success=sum(r['direction'] == d and r['success_10s'] for r in rows))
                               for d in sorted({r['direction'] for r in rows})}
    return result


def main():
    import mujoco
    import yaml
    p = argparse.ArgumentParser()
    p.add_argument('--cases', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--workers', type=int, default=8)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    cases = json.loads(a.cases.read_text())
    assert len(cases) == 64
    stress = [c for d in ('supine', 'prone', 'left_side_down', 'right_side_down')
              for c in [c for c in cases if c['direction'] == d][:4]]
    cfg = yaml.safe_load((ROOT / 'policy/smp_recovery/config/smp_recovery.yaml').read_text())
    amp_cfg = yaml.safe_load((ROOT / 'policy/amp/config/amp.yaml').read_text())
    profiles = ('ft_r2_9000', 'path_a6_9999', 'm2_9999')
    models = {p: digest(ROOT / 'policy/smp_recovery/model' / cfg['model_profiles'][p]['model_path']) for p in profiles}
    models['amp'] = digest(ROOT / 'policy/amp/model' / amp_cfg['model_path'])
    protocol = dict(mujoco=mujoco.__version__, seconds=20, physics_dt=.002, control_dt=.02,
                    cases_sha256=digest(a.cases), model_sha256=models,
                    robot_xml_sha256=digest(ROOT / 'g1_description/g1_liao.xml'),
                    config_sha256=digest(ROOT / 'policy/smp_recovery/config/smp_recovery.yaml'),
                    amp_config_sha256=digest(ROOT / 'policy/amp/config/amp.yaml'),
                    runner_sha256=digest(__file__), controller_eval_sha256=digest(ROOT / 'tools/validate_ft12k_sim.py'),
                    scenarios=SCENARIOS, stress_cases=[c['name'] for c in stress],
                    upper_definition='torso_link and descendants, mass and inertia scaled together',
                    gain_definition='physical Kp and Kd scaled together, host target limiter unchanged',
                    delay_definition='constant physical position-command delay at 2ms resolution; nominal host observations',
                    load_definition='500Hz actual actuator torque and joint velocity; mechanical abs(tau*dq)',
                    scope='offline deployment controller, procedural flat cases; no obstacles, push, DDS, hardware or random noise')
    (a.out / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    jobs = []
    for scenario in SCENARIOS:
        selected = cases if scenario == 'nominal' else stress
        for profile in (*profiles, 'amp') if scenario == 'nominal' else profiles:
            folder = a.out / scenario / profile
            folder.mkdir(parents=True)
            (folder / 'cases.json').write_text(json.dumps(selected) + '\n')
            for i in range(0, len(selected), 4):
                jobs.append((profile, scenario, selected[i:i+4], str(folder)))
    rows = []
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context('spawn')) as pool:
        for f in as_completed([pool.submit(worker, job) for job in jobs]):
            rows.extend(f.result())
            print('COMPLETE', len(rows), '/', sum(len(j[2]) for j in jobs), flush=True)
    summary = {s: {p: aggregate([r for r in rows if r['scenario'] == s and r['profile'] == p])
                   for p in (*profiles, 'amp') if any(r['scenario'] == s and r['profile'] == p for r in rows)}
               for s in SCENARIOS}
    (a.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print('VALIDATION_COMPLETE', a.out, flush=True)


if __name__ == '__main__':
    main()
