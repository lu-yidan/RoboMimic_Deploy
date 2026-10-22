"""Compare a geometry-observation wrapper with complete audited zero trajectories."""
import argparse
import json
from pathlib import Path
import numpy as np
from training.strict_full_cycle_residual_env import FullCycleResidualEnv
from training.bar_geometry_observation import BarGeometryObservation, ROWS
from deploy_mujoco.run_trained_full_cycle import sha


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--parent',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    m=json.loads((args.parent/'manifest.json').read_text())
    audit=json.loads((args.parent/'independent_training_audit.json').read_text())
    summary=json.loads((args.parent/'summary.json').read_text())
    if not audit['passed'] or sha(args.parent/'summary.json')!=audit['experiment_summary_sha256']:
        raise ValueError('Audited parent required')
    args.output.mkdir(exist_ok=False)
    results=[]
    for case in ('nominal','foot_z_plus10'):
        record=next(r for r in summary['zero'] if r['case']==case)
        path=args.parent/f'zero_{case}.npz'
        if sha(path)!=record['trajectory_sha256']:raise ValueError('Saved trajectory changed')
        for mode in ('blind','oracle'):
            base=FullCycleResidualEnv(plan=m['reference'],cases={case:m['evaluation_cases'][case]},
                normal_bias_y_deg=m['normal_bias_y_deg'],memory_enabled=m['force_history_memory'])
            env=BarGeometryObservation(base,mode=mode)
            expected=np.array([m['evaluation_cases'][case].get(row,[0.,0.,0.]) for row in ROWS]).reshape(-1)/.02
            if mode=='blind':expected[:]=0.
            try:
                obs,_=env.reset(options={'case':case})
                with np.load(path) as z:
                    assert np.array_equal(base.env.data.qpos,z['qpos'][0])
                    for i in range(len(z['actions'])):
                        assert np.array_equal(obs[:136],z['observations'][i])
                        np.testing.assert_allclose(obs[136:],expected,rtol=0,atol=1e-7)
                        obs,reward,terminated,truncated,info=env.step(z['actions'][i])
                        assert reward==z['rewards'][i]
                        for field in ('qpos','qvel'):
                            assert np.array_equal(getattr(base.env.data,field),z[field][i+1])
                        assert base.env.data.time==z['time_s'][i+1]
                        assert np.array_equal(base.last_torque,z['torques'][i])
                        assert (terminated or truncated)==(i==len(z['actions'])-1)
                    assert info['reason']==record['reason'] and info['success']==record['success']
                row=dict(case=case,mode=mode,time_s=info['time_s'],original_success=info['success'],
                    exact_replay=True,trajectory_sha256=sha(path))
                results.append(row);print(json.dumps(row),flush=True)
            finally:env.close()
    sources=[__file__,'training/bar_geometry_observation.py','training/strict_full_cycle_residual_env.py']
    result=dict(passed=True,results=results,parent_audit_sha256=sha(args.parent/'independent_training_audit.json'),
        sources={p:sha(p) for p in sources},scope='Exact complete zero-policy trajectory reproduction in blind/oracle modes: nominal success and early collision failure. No learned148-input policy evaluated.')
    (args.output/'summary.json').write_text(json.dumps(result,indent=2))
    for p in sources:(args.output/Path(p).name).write_bytes(Path(p).read_bytes())


if __name__=='__main__':main()
