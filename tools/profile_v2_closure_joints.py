"""Locate closure deviations without conflating them with policy target offsets."""
import argparse
import json
from pathlib import Path
import numpy as np
from training.wall_support_env import WallSupportEnv
from tools.validate_swing_reference import read,write,sha


def profile(run):
    m=read(run/'manifest.json');audit=read(run/'v2_multicycle_audit.json');s=read(run/'summary.json')
    assert audit['passed'] and audit['manifest_sha256']==sha(run/'manifest.json')
    assert audit['trajectory_sha256']==sha(run/'trajectory.npz')
    with np.load(run/'trajectory.npz') as f:z={k:f[k] for k in ('qpos','time_s','raw_observations','actions')}
    with np.load(run/'reference.npz') as f:r={k:f[k] for k in ('qpos','time','cycle_ends_s')}
    e=WallSupportEnv(Path(m['bundle']));action_joints=np.r_[np.arange(12),np.arange(15,29)]
    rows=[]
    try:
        for i,end in enumerate(r['cycle_ends_s']):
            ids=np.flatnonzero((z['time_s']>end-2)&(z['time_s']<=end+1e-7))
            assert len(ids)>=99
            error=z['qpos'][ids][:,e.qadr]-r['qpos'][0,e.qadr]
            reference_index=int(np.argmin(abs(r['time']-end)));joints=[]
            for j,joint_id in enumerate(e.joints):
                k=int(np.argmax(abs(error[:,j])));state_index=int(ids[k])
                action_index=np.flatnonzero(action_joints==j)
                observed=state_index<len(z['raw_observations']) and len(action_index)==1
                residual=None if not observed else float(z['raw_observations'][state_index,110+int(action_index[0])]*.015*1000)
                joints.append(dict(joint=e.model.joint(int(joint_id)).name,
                    peak_absolute_closure_mrad=float(abs(error[k,j])*1000),
                    signed_closure_at_peak_mrad=float(error[k,j]*1000),time_at_peak_s=float(z['time_s'][state_index]),
                    previous_filtered_policy_target_offset_mrad=residual,
                    nominal_final_reference_closure_mrad=float((r['qpos'][reference_index,e.qadr[j]]-r['qpos'][0,e.qadr[j]])*1000)))
            joints.sort(key=lambda v:v['peak_absolute_closure_mrad'],reverse=True)
            assert abs(joints[0]['peak_absolute_closure_mrad']-s['cycle_results'][i]['closure_mrad'])<1e-9
            rows.append(dict(cycle=i+1,joints=joints,global_closure_matches_runtime=True))
    finally:e.close()
    return dict(run=str(run),case=m['case'],reference_sha256=m['plan_sha256'],checkpoint_sha256=m['checkpoint_sha256'],
        audit_sha256=sha(run/'v2_multicycle_audit.json'),trajectory_sha256=sha(run/'trajectory.npz'),cycles=rows)


def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();baseline=profile(a.baseline.resolve());candidate=profile(a.candidate.resolve())
    assert all(baseline[k]==candidate[k] for k in ('case','reference_sha256','checkpoint_sha256'))
    a.output.mkdir(parents=True,exist_ok=False)
    write(a.output/'joint_profile.json',dict(baseline=baseline,candidate=candidate,source_sha256=sha(__file__),
        scope='Maximum deviation from the initial joint posture over each final two-second interval. Nominal endpoint reference displacement is separately listed; first-cycle reference modifiers are not decomposed here. Policy offsets come from the prior-command filtered residual stored in observations, with float32 precision; they are not motor torques. This localization is observational, not a causal attribution of the whole-body error. No new training or controller changes.'))
    (a.output/'analysis_source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({label:[dict(cycle=r['cycle'],dominant=r['joints'][0]) for r in value['cycles']]
                      for label,value in [('baseline',baseline),('candidate',candidate)]}),flush=True)


if __name__=='__main__':main()
