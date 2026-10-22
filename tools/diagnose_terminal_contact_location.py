"""Reconstruct geometric contact locations from saved states, without dynamics."""
import argparse
import json
from pathlib import Path
import mujoco
import numpy as np
from training.strict_full_cycle_residual_env import FullCycleResidualEnv
from tools.continue_vector_full_cycle_training import sha


def main():
    parser=argparse.ArgumentParser();parser.add_argument('experiment',type=Path);args=parser.parse_args();rows=[]
    for record in json.loads((args.experiment/'results.json').read_text()):
        p=args.experiment/record['label'];m=json.loads((p/'manifest.json').read_text())
        audit=json.loads((p/'independent_audit.json').read_text())
        assert audit['integrity_passed'] and audit['trajectory_sha256']==sha(p/'trajectory.npz')
        env=FullCycleResidualEnv(plan=m['reference']);e=env.env
        try:
            env.cases={m['case']:m['offsets']};env.reset(options={'case':m['case']});model=e.model;d=mujoco.MjData(model)
            soles={model.geom(side+'_sole_surface').id:side for side in ('left','right')}
            contacts={side:[] for side in ('left','right')};positions=[]
            with np.load(p/'trajectory.npz') as z:
                ids=np.flatnonzero(z['time_s']>env.horizon-2)
                for i in ids:
                    d.qpos[:]=z['qpos'][i];mujoco.mj_kinematics(model,d);mujoco.mj_collision(model,d)
                    positions.append(d.xpos[env.endpoint_bodies].copy())
                    for c in d.contact:
                        a,b=int(c.geom1),int(c.geom2)
                        if a not in soles and b not in soles:continue
                        sole=a if a in soles else b;other=b if sole==a else a
                        name=model.geom(other).name or ''
                        if 'rung_01' not in name:continue
                        body=int(model.geom_bodyid[sole]);local=d.xmat[body].reshape(3,3).T@(c.pos-d.xpos[body])
                        contacts[soles[sole]].append([float(local[0]),float(c.dist)])
            stats={}
            for side,values in contacts.items():
                array=np.asarray(values)
                stats[side]=dict(count=len(values),local_x_min_m=float(array[:,0].min()),local_x_max_m=float(array[:,0].max()),local_x_mean_m=float(array[:,0].mean()),distance_min_m=float(array[:,1].min())) if values else dict(count=0)
            d.qpos[:]=env.plan['qpos'][-1];mujoco.mj_kinematics(model,d)
            errors=np.max(np.linalg.norm(np.asarray(positions)-d.xpos[env.endpoint_bodies],axis=2),axis=0)
            rows.append(dict(case=m['case'],contacts=stats,terminal_body_error_m=errors.tolist(),trajectory_sha256=sha(p/'trajectory.npz'),audit_sha256=sha(p/'independent_audit.json')))
        finally:env.close()
    result=dict(rows=rows,source_sha256=sha(__file__),scope='Collision geometry reconstructed from final two seconds. Contact points are not measured force or evidence of force distribution; no state is injected into a running experiment.')
    (args.experiment/'contact_location_diagnostic.json').write_text(json.dumps(result,indent=2));(args.experiment/'contact_location_diagnostic_source.py').write_bytes(Path(__file__).read_bytes());print(json.dumps(result,indent=2))


if __name__=='__main__':main()
