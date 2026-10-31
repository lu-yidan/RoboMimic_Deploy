"""Small declared common-command feasibility search, not learned feedback."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import linprog
from training.closure_legtrim_curriculum_env import ClosureLegtrimCurriculumEnv
from tools.compare_audited_closure_candidates import load_run,sha

CASES=['foot_z_plus15','z15.1','x-0.1_z15','x+0.1_z15']


def main():
    parser=argparse.ArgumentParser()
    for name in ('parent','cache','env-gate','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(1)
    verified=load_run(args.parent);manifest=json.loads((args.parent/'manifest.json').read_text())
    gate=json.loads((args.env_gate/'summary.json').read_text());assert gate['passed'] and gate['teacher_sha256']==sha(args.parent/'policy.zip')
    for name,value in gate['sources'].items():assert sha(name)==value
    assert gate['cache_summary_sha256']==sha(args.cache/'summary.json')
    args.output.mkdir(exist_ok=False)
    protocol=dict(parent=str(args.parent.resolve()),parent_audit_sha256=verified['audit_sha256'],cache_summary_sha256=sha(args.cache/'summary.json'),
        environment_gate_sha256=sha(args.env_gate/'summary.json'),cases=CASES,finite_difference=.25,proposal_radius=1.,proposal_fractions=[.25,.5,.75,1.],
        coordinates=['left_knee','left_ankle_pitch'],source_sha256=sha(__file__),
        scope='Same two constant commands across four known difficult development cases. Zero, four signed probes, then four fractions of one linearized minimax proposal. Extra knee/ankle reference authority up to 0.003 rad; original actuator/task limits retained. No actor training, no held-out claim, no default replacement.')
    (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2));(args.output/'search_source.py').write_bytes(Path(__file__).read_bytes())
    env=ClosureLegtrimCurriculumEnv(args.cache,controller_modifier=manifest['controller_modifier'],teacher_checkpoint=args.parent/'policy.zip',teacher_sha256=sha(args.parent/'policy.zip'))
    rows=[];constraints=[]

    def evaluate(command,label):
        directory=args.output/f'trial_{len(rows):02d}_{label}';directory.mkdir()
        case_rows=[];vector=[]
        for case in CASES:
            _,_=env.reset(options={'case':case});base=env.env.unwrapped
            q=[base.env.data.qpos.copy()];v=[base.env.data.qvel.copy()];times=[float(base.env.data.time)]
            while True:
                _,_,done,truncated,info=env.step(command)
                q.append(base.env.data.qpos.copy());v.append(base.env.data.qvel.copy());times.append(float(base.env.data.time))
                if done or truncated:break
            q=np.asarray(q);v=np.asarray(v);times=np.asarray(times);tail=times>111.66
            complete=bool(len(q)==701 and tail.sum()==101)
            joint=q[tail][:,base.env.qadr]-base.env.initial[base.env.qadr]
            height=q[tail,2]-base.env.initial[2]-.25
            signed=np.r_[joint.reshape(-1)/.001,-joint.reshape(-1)/.001,height/.0001,-height/.0001] if complete else None
            bounds=np.r_[np.full(2*joint.size,40.),np.full(2*len(height),100.)] if complete else None
            margin=float((bounds-signed).min()) if complete else -1e9
            if label=='zero':
                with np.load(args.parent/f'after_{case}.npz') as original:
                    assert np.max(abs(original['qpos'][4983:]-q))<1e-10 and np.max(abs(original['qvel'][4983:]-v))<1e-8
            path=directory/f'{case}.npz';np.savez_compressed(path,qpos=q,qvel=v,time_s=times,command=command)
            case_rows.append(dict(case=case,success=bool(info['success']),reason=info['reason'],complete=complete,margin_score=margin,
                closure_rad=float(abs(joint).max()) if len(joint) else None,height_error_m=float(abs(height).max()) if len(height) else None,trajectory_sha256=sha(path)))
            vector.append((signed,bounds))
        row=dict(index=len(rows),label=label,command=np.asarray(command).tolist(),cases=case_rows,
            successes=sum(r['success'] for r in case_rows),highest_preserved=case_rows[0]['success'],minimum_margin=min(r['margin_score'] for r in case_rows))
        rows.append(row);constraints.append(vector);(directory/'summary.json').write_text(json.dumps(row,indent=2))
        print(json.dumps(row),flush=True)
        return len(rows)-1

    try:
        evaluate(np.zeros(2),'zero')
        for axis in range(2):
            for sign in (-1,1):
                command=np.zeros(2);command[axis]=sign*.25;evaluate(command,f'axis{axis}_{sign:+d}')
        valid=all(all(r['complete'] for r in row['cases']) for row in rows)
        proposal=None
        if valid:
            center=np.concatenate([v[0] for v in constraints[0]]);bounds=np.concatenate([v[1] for v in constraints[0]])
            jacobian=np.stack([(np.concatenate([v[0] for v in constraints[2+2*k]])-np.concatenate([v[0] for v in constraints[1+2*k]]))/.5 for k in range(2)],axis=1)
            solved=linprog([0,0,-1],A_ub=np.c_[jacobian,np.ones(len(center))],b_ub=bounds-center,bounds=[(-1,1)]*2+[(None,None)],method='highs')
            proposal=dict(success=bool(solved.success),message=solved.message)
            if solved.success:
                proposal.update(command=solved.x[:2].tolist(),predicted_minimum_margin=float(solved.x[2]))
                for fraction in protocol['proposal_fractions']:evaluate(solved.x[:2]*fraction,f'linear_{fraction}')
        best=max(rows,key=lambda r:(r['highest_preserved'],r['successes'],r['minimum_margin']))
        result=dict(completed=True,linearization_available=valid,proposal=proposal,rows=rows,best=best,
            protocol_sha256=sha(args.output/'protocol.json'),scope=protocol['scope'])
        (args.output/'summary.json').write_text(json.dumps(result,indent=2))
    finally:env.close()


if __name__=='__main__':main()
