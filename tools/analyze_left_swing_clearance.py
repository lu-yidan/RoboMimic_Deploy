"""Compare scripted and recorded foot clearance without stepping dynamics."""
import argparse
import json
from pathlib import Path
import mujoco
import numpy as np
from training.strict_full_cycle_residual_env import FullCycleResidualEnv
from tools.compare_audited_closure_candidates import load_run,sha


def main():
    p=argparse.ArgumentParser()
    for key in ('failure','successful','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();paths=[a.successful,a.failure];cases=['foot_z_plus15','fresh_boundary_03']
    runs=[load_run(path) for path in paths]
    assert runs[0]['reference_sha256']==runs[1]['reference_sha256']
    assert runs[0]['checkpoint_sha256']==runs[1]['checkpoint_sha256']
    assert runs[0]['adaptive_reference_command']==runs[1]['adaptive_reference_command']
    assert runs[0]['rows'][cases[0]]['success'] and not runs[1]['rows'][cases[1]]['success']
    arrays={};rows=[]
    for label,path,case in zip(('successful','failure'),paths,cases):
        m=json.loads((path/'manifest.json').read_text())
        env=FullCycleResidualEnv(plan=m['reference'],cases={case:m['evaluation_cases'][case]})
        env.reset(options={'case':case});e=env.env;data=mujoco.MjData(e.model)
        sole=e.model.geom('left_sole_surface').id;rod=e.model.geom('rung_01').id
        def distance(q):
            data.qpos[:]=q;mujoco.mj_kinematics(e.model,data)
            return float(mujoco.mj_geomDistance(e.model,data,sole,rod,1.,np.zeros(6)))
        try:
            times=np.arange(70.2,80.6600001,.02)
            reference=np.asarray([distance(env.reference(float(t))[0]) for t in times])
            with np.load(path/f'after_{case}.npz') as z:
                mask=(z['time_s']>=70.2-1e-7)&(z['time_s']<=80.66+1e-7)
                recorded_times=z['time_s'][mask];actual=np.asarray([distance(q) for q in z['qpos'][mask]])
            arrays[label+'_reference_time_s']=times;arrays[label+'_reference_distance_m']=reference
            arrays[label+'_recorded_time_s']=recorded_times;arrays[label+'_recorded_distance_m']=actual
            rows.append(dict(case=case,offsets=m['evaluation_cases'][case],
                reference_minimum_distance_m=float(reference.min()),reference_minimum_time_s=float(times[reference.argmin()]),
                recorded_minimum_distance_m=float(actual.min()),recorded_minimum_time_s=float(recorded_times[actual.argmin()]),
                recorded_until_s=float(recorded_times[-1]),trajectory_sha256=sha(path/f'after_{case}.npz')))
        finally:env.close()
    a.output.mkdir(exist_ok=False);np.savez_compressed(a.output/'clearance.npz',**arrays)
    result=dict(rows=rows,data_sha256=sha(a.output/'clearance.npz'),source_sha256=sha(__file__),
        scope='Offline foot/target-rod geometry. Dashed reference is the scripted pose before actor/controller additions; solid actual is recorded physics. No dynamics replay, complete-body collision certificate or isolated causal claim. Failed trajectory ends at its actual termination time.')
    (a.output/'summary.json').write_text(json.dumps(result,indent=2));(a.output/'source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(result))


if __name__=='__main__':main()
