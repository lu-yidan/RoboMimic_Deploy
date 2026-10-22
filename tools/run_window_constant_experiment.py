"""Six full cycles: matched biased load window, memory disabled/enabled."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def trial(item):
    from omegaconf import OmegaConf
    from deploy_mujoco.run_window_constant_cycle import run_climb_cycle
    out,case,offsets,memory=item;name='constant__'+case
    run=out/name
    cfg=OmegaConf.load(ROOT/'deploy_mujoco/config/mujoco_climb_cycle_optimized.yaml').climb_cycle
    cfg.output_dir=str(run);cfg.rung_offsets=offsets
    cfg.plan=str(ROOT/'outputs/clearance_back20_leg14_plan_v1/plan.npz')
    cfg.contact_frame_from_transfer=True;cfg.reset_contact_frame_each_cycle=True
    cfg.window_memory_enabled=memory;cfg.window_normal_bias_y_deg=-3.
    with (out/(name+'.log')).open('w') as stream:
        saved=sys.stdout;sys.stdout=stream
        try:run_climb_cycle(cfg)
        finally:sys.stdout=saved
    result=json.loads((run/'summary.json').read_text())
    checks={}
    for tool,filename in [('audit_climb_cycle.py','independent_audit.json'),('audit_joint_limits.py','joint_limit_audit.json')]:
        process=subprocess.run([sys.executable,str(ROOT/'tools'/tool),'--run',str(run)],capture_output=True,text=True)
        (run/(tool+'.log')).write_text(process.stdout+process.stderr)
        checks[filename]=json.loads((run/filename).read_text())['passed'] if (run/filename).exists() else False
    return dict(case=case,memory_enabled=memory,run=name,passed=bool(result['passed'] and all(checks.values())),reason=result['reason'],time_s=result['simulation_time_s'],checks=checks,minimum_native_joint_margin_rad=result['minimum_native_joint_margin_rad'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--workers',type=int,default=2);a=p.parse_args()
    a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=False)
    cases=[('nominal',{}),('row01_z_+5mm',{'rung_01':[0.,0.,.005]}),('row01_x_-10mm',{'rung_01':[-.01,0.,0.]})]
    protocol=dict(cases=cases,normal_bias_y_deg=-3.,window_s=[49.,61.56],memory_options=[False],constant_action=[-.25,0,0,0,0,0],residual_scale_rad=.015,filter_alpha=.05,filter_dt_s=.005,scope='Continuous from0; only49..61.56s normals biased. No fallback to baseline during intervention if support disappears. Outside window original early contact establishment. Constant action selected on one known hardcase, ramps to zero after61.56s via same filter; not learned. Not full-cycle sensor-bias robustness; single-cycle nominal reference only.')
    (a.out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
    results=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for result in pool.map(trial,[(a.out,case,offsets,memory) for memory in [False] for case,offsets in cases]):
            results.append(result);(a.out/'results.json').write_text(json.dumps(results,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
