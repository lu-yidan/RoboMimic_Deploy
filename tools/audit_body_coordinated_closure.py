"""Reconstruct bounded feedback, physical margins and endpoint gates from records."""
import argparse
import json
from pathlib import Path
import mujoco
import numpy as np
import torch
from deploy_mujoco.run_geometry_full_cycle import load_verified
from tools.audit_multirung_cycle import audit_endpoints
from tools.continue_vector_full_cycle_training import sha


def audit(p):
    prov=json.loads((p/'provenance.json').read_text())
    summary=json.loads((p/'summary.json').read_text())
    base, observer, model, manifest, _=load_verified(Path('outputs/closure_margin_control98304_v1'),prov['case'])
    observer.reset(options={'case':prov['case']});e=base.env
    try:
        with np.load(p/'trajectory.npz') as saved:z={k:saved[k].copy() for k in saved.files}
        ts=z['time_s'][:-1]
        u=np.clip((ts-99.66)/3.,0.,1.)
        errors=z['qpos'][:-1,e.qadr[base.indices]]-base.plan['qpos'][0,e.qadr[base.indices]]
        feedback=np.zeros_like(z['actions']);feedback[:,:12]=np.clip(-prov['gain']*errors[:,:12]/.015,-1.,1.)
        predicted=np.clip(z['actor_actions']+(u*u*(3-2*u))[:,None]*feedback,-1.,1.)
        endpoints=audit_endpoints(e,z,base.plan)
        margin=float(np.minimum(z['qpos'][:,e.qadr]-e.model.jnt_range[e.joints,0],e.model.jnt_range[e.joints,1]-z['qpos'][:,e.qadr]).min())
        scratch=mujoco.MjData(e.model);penetration=0.
        for q in z['qpos']:
            scratch.qpos[:]=q;mujoco.mj_kinematics(e.model,scratch);mujoco.mj_collision(e.model,scratch)
            for contact in scratch.contact:
                a,b=[int(e.model.geom_bodyid[g]) for g in (contact.geom1,contact.geom2)]
                ina,inb=a in e.robot_bodies,b in e.robot_bodies
                if not (ina or inb):continue
                expected=ina!=inb and (a if ina else b) in e.contact_groups
                if not expected:penetration=max(penetration,float(-contact.dist))
        tail=(z['time_s']>base.horizon-2.)&(z['time_s']<=base.horizon+1e-7)
        rise=z['qpos'][tail,2]-base.plan['qpos'][0,2]
        checks=dict(trajectory_hash=sha(p/'trajectory.npz')==summary['trajectory_sha256'],
            frozen_runner=sha(p.parent/'runner_source.py')==prov['source_sha256'],
            protocol_hash=sha(p.parent/'protocol.json')==prov['protocol_sha256'],
            body_target_source=sha(p.parent/'body_target_source.py')==prov['body_target_source_sha256'],
            parent_hash=sha(Path('outputs/closure_margin_control98304_v1/policy.zip'))==prov['parent_checkpoint_sha256'],
            reference_hash=sha(manifest['reference'])==prov['reference_sha256'],
            finite=all(np.isfinite(a).all() for a in z.values()),
            action_reconstruction=bool(np.allclose(predicted,z['actions'],atol=1e-14,rtol=0)),
            action_bounds=bool(abs(z['actions']).max()<=1.),
            torque_bounds=bool(np.all(z['torques']<=e.limits[:,1]+1e-8) and np.all(z['torques']>=e.limits[:,0]-1e-8)),
            native_limits=margin>=0.,unexpected_penetration=penetration<1e-5,
            endpoint_reconstruction=endpoints==summary['endpoints'])
        accepted=bool(all(checks.values()) and summary['runtime']['success'] and all(r['passed'] for r in endpoints)
            and len(rise)>=99 and np.all(abs(rise-.25)<.01))
        result=dict(integrity_passed=all(checks.values()),independently_accepted=accepted,checks=checks,
            endpoints=endpoints,minimum_sampled_native_margin_rad=margin,unexpected_penetration_m=penetration,
            trajectory_sha256=sha(p/'trajectory.npz'),summary_sha256=sha(p/'summary.json'),
            audit_source_sha256=sha(__file__),scope='Independent saved-state/action/endpoints check; substep force/contact-memory checks remain those enforced by the unchanged live environment. Not a separate policy replay.')
        (p/'independent_audit.json').write_text(json.dumps(result,indent=2))
        return dict(run=p.name,accepted=accepted,integrity_passed=result['integrity_passed'],joint_closure=endpoints[0]['joint_closure_rad'])
    finally:observer.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('suite',type=Path);args=p.parse_args();torch.set_num_threads(1)
    protocol=json.loads((args.suite/'protocol.json').read_text())
    rows=[audit(args.suite/f'gain{gain:g}_drop{drop:g}_{case}') for gain in protocol['gains'] for drop in protocol['drops'] for case in protocol['cases']]
    (args.suite/'independent_audits.json').write_text(json.dumps(rows,indent=2))
    (args.suite/'audit_source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
