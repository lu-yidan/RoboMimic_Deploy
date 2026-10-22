"""Full saved-trajectory equality gate for the optional margin cost wrapper."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import numpy as np
from training.strict_full_cycle_residual_env import FullCycleResidualEnv
from training.bar_geometry_observation import BarGeometryObservation
from training.closure_margin_reward import ClosureMarginReward
from tools.continue_vector_full_cycle_training import sha


def check(label):
    p=Path('outputs/support_retarget_calibration_v1')/label
    m=json.loads((p/'manifest.json').read_text());s=json.loads((p/'summary.json').read_text());a=json.loads((p/'independent_audit.json').read_text())
    assert a['integrity_passed'] and a['trajectory_sha256']==sha(p/'trajectory.npz')
    base=FullCycleResidualEnv(plan=m['reference'],cases={m['case']:m['offsets']},normal_bias_y_deg=-3.)
    env=ClosureMarginReward(BarGeometryObservation(base,mode='oracle'),weight=.02)
    obs,_=env.reset(options={'case':m['case']});e=base.env
    maxima=dict(qpos=0.,qvel=0.,torques=0.,base_observation=0.,base_reward=0.,reward_identity=0.)
    costs=[]
    with np.load(p/'trajectory.npz') as archive:
        z={key:archive[key] for key in archive.files}
        assert np.array_equal(e.data.qpos,z['qpos'][0]) and np.array_equal(e.data.qvel,z['qvel'][0])
        for i,action in enumerate(z['actions']):
            maxima['base_observation']=max(maxima['base_observation'],float(abs(obs[:136]-z['observations'][i]).max()))
            obs,reward,done,truncated,info=env.step(action)
            for key,array in [('qpos',e.data.qpos),('qvel',e.data.qvel)]:maxima[key]=max(maxima[key],float(abs(array-z[key][i+1]).max()))
            maxima['torques']=max(maxima['torques'],float(abs(base.last_torque-z['torques'][i]).max()))
            maxima['base_reward']=max(maxima['base_reward'],abs(info['base_reward']-float(z['rewards'][i])))
            q=z['qpos'][i+1];closure=float(abs(q[e.qadr]-z['qpos'][0,e.qadr]).max());rise=float(q[2]-z['qpos'][0,2])
            u=max(0.,min(1.,(float(z['time_s'][i+1])-99.66)/3.));ramp=u*u*(3-2*u)
            joint=max(0.,min(1.,(closure-.032)/.008));height=max(0.,min(1.,(abs(rise-.25)-.006)/.004))
            expected=.01*ramp*(joint*joint+height*height)
            maxima['reward_identity']=max(maxima['reward_identity'],abs(reward-(float(z['rewards'][i])-expected)))
            costs.append(expected)
            assert abs(e.data.time-z['time_s'][i+1])<1e-10 and not truncated and done==(i==len(z['actions'])-1)
    expected_extra=np.zeros(12);expected_extra[2]=m['offsets']['rung_01'][2]/.02
    checks=dict(exact_physics_and_original_observation=all(maxima[k]==0. for k in ('qpos','qvel','torques','base_observation','base_reward')),
        reward_identity=maxima['reward_identity']<1e-12,geometry_channels=bool(np.allclose(obs[136:],expected_extra,atol=1e-7,rtol=0)),
        unchanged_termination=info['reason']==s['reason'] and info['success']==s['success'],unchanged_runtime_peaks=info['peaks']==s['peaks'],
        bounded_cost=all(0<=x<=.02 for x in costs),nonzero_cost=any(x>0 for x in costs))
    env.close()
    return dict(label=label,passed=all(checks.values()),checks=checks,maxima=maxima,samples=len(costs),cost_sum=sum(costs),source_trajectory_sha256=sha(p/'trajectory.npz'))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();args.output.mkdir(exist_ok=False)
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
    sources={p:sha(p) for p in ('training/closure_margin_reward.py','training/bar_geometry_observation.py','training/strict_full_cycle_residual_env.py','training/full_cycle_residual_env.py',__file__)}
    (args.output/'protocol.json').write_text(json.dumps(dict(cases=['true_height_12.5mm','true_height_15mm'],sources=sources),indent=2))
    with ProcessPoolExecutor(max_workers=2) as pool:rows=list(pool.map(check,['true_height_12.5mm','true_height_15mm']))
    result=dict(passed=all(r['passed'] for r in rows),results=rows,sources=sources,scope='Full successful and failed zero-action dynamics identical to audited records; only bounded training reward changes. No learning benefit established.')
    (args.output/'summary.json').write_text(json.dumps(result,indent=2));(args.output/'gate_source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(result,indent=2))
    if not result['passed']:raise AssertionError(result)


if __name__=='__main__':main()
