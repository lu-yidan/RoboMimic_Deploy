"""Predeclared deterministic, one-factor-at-a-time rung-error experiment."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def run_case(item):
    from omegaconf import OmegaConf
    from deploy_mujoco.run_early_contact_cycle import run_climb_cycle
    name,offsets,guard,out,*reference=item
    cfg=OmegaConf.load(ROOT/'deploy_mujoco/config/mujoco_climb_cycle_optimized.yaml').climb_cycle
    cfg.output_dir=str(out/name);cfg.rung_offsets=offsets;cfg.release_guard=guard
    if reference:cfg.plan=str(reference[0])
    if len(reference)>1:cfg.allocation_failure_hold=bool(reference[1])
    if len(reference)>2:cfg.unload_lookahead_seconds=float(reference[2])
    if len(reference)>3:cfg.tracking_relaxation=bool(reference[3])
    if len(reference)>4:cfg.contact_frame_controller=bool(reference[4])
    if len(reference)>5:cfg.contact_frame_transfer=bool(reference[5])
    if len(reference)>6:cfg.contact_frame_from_transfer=bool(reference[6])
    if len(reference)>7:cfg.contact_landing=bool(reference[7])
    if len(reference)>8:cfg.reset_contact_frame_each_cycle=bool(reference[8])
    log=out/(name+'.log')
    with log.open('w') as f:
        import contextlib
        with contextlib.redirect_stdout(f):
            run_climb_cycle(cfg)
    result=subprocess.run([sys.executable,str(ROOT/'tools/audit_climb_cycle.py'),'--run',str(out/name)],capture_output=True,text=True)
    (out/name/'audit_console.log').write_text(result.stdout+result.stderr)
    s=json.loads((out/name/'summary.json').read_text())
    audit=json.loads((out/name/'independent_audit.json').read_text()) if result.returncode==0 else {'passed':False}
    landing_audit={'passed':True,'applicable':False}
    if cfg.get('contact_landing',False):
        extra=subprocess.run([sys.executable,str(ROOT/'tools/audit_contact_landing.py'),'--run',str(out/name)],capture_output=True,text=True)
        (out/name/'landing_audit_console.log').write_text(extra.stdout+extra.stderr)
        landing_audit=json.loads((out/name/'landing_audit.json').read_text()) if (out/name/'landing_audit.json').exists() else {'passed':False}
    return dict(name=name,offsets=offsets,guard=guard,passed=s['passed'] and audit['passed'] and landing_audit['passed'],
                reason=s['reason'],time_s=s['simulation_time_s'],checks=s['checks'],
                base_error_m=s['base_error_peak_m'],other_contact_n=s['other_contact_peak_n'],
                flight_contact_n=s['flight_contact_peak_n'],self_contact_n=s['self_contact_peak_n'],
                final=s['final'],release_blocked_at_s=s['release_blocked_at_s'],
                hold_duration_s=s['hold_duration_s'],audit=audit,landing_audit=landing_audit)


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--workers',type=int,default=3);p.add_argument('--guard',action='store_true')
    p.add_argument('--failure-hold',action='store_true',help='Try confirmed-support current-pose hold after tracking allocation failure; requires --guard')
    p.add_argument('--lookahead-ms',type=float,default=0.,help='Privileged-model left-unload feasibility prediction, every 50 ms')
    p.add_argument('--relax-tracking',action='store_true',help='On infeasible allocation reduce tracking demand, retaining original force and motor bounds')
    p.add_argument('--contact-frame',action='store_true',help='Oracle: actual MuJoCo contact points and normal-frame friction cones')
    p.add_argument('--contact-frame-transfer',action='store_true',help='Oracle contact frames only during weight transfer and left unload/swing/landing; baseline establishes contacts')
    p.add_argument('--contact-frame-from-transfer',action='store_true',help='Oracle contact frames from the first weight transfer through the end of the cycle')
    p.add_argument('--contact-landing',action='store_true',help='Bounded LF search and probe-loading; requires continued contact frames; separate online audit required')
    p.add_argument('--reset-contact-frame-each-cycle',action='store_true',help='Establish hand/foot contacts with baseline again in every cycle; requires --contact-frame-from-transfer')
    p.add_argument('--plan',type=Path,default=ROOT/'outputs/compact_cycle_plan_v13/plan.npz')
    p.add_argument('--stress',action='store_true',help='Add separately labelled +/-15 mm hand-target stress cases')
    p.add_argument('--cases',nargs='*');a=p.parse_args()
    if a.failure_hold and not a.guard:raise ValueError('--failure-hold requires --guard')
    if a.lookahead_ms and not a.failure_hold:raise ValueError('--lookahead-ms requires --failure-hold')
    a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=False)
    cases=[('nominal',{})]
    for row in ('01','05'):
        for axis,index in [('x',0),('z',2)]:
            for mm in (-10,-5,5,10):
                offset=[0.,0.,0.];offset[index]=mm/1000
                cases.append((f'row{row}_{axis}_{mm:+d}mm',{'rung_'+row:offset}))
    if a.stress:
        for axis,index in [('x',0),('z',2)]:
            for mm in (-15,15):
                offset=[0.,0.,0.];offset[index]=mm/1000
                cases.append((f'row05_{axis}_{mm:+d}mm',{'rung_05':offset}))
    if a.cases:cases=[c for c in cases if c[0] in a.cases]
    if not cases:raise ValueError('No selected cases')
    protocol=dict(cases=cases,release_guard=a.guard,allocation_failure_hold=a.failure_hold,tracking_relaxation=a.relax_tracking,
                  contact_frame_oracle=a.contact_frame,
                  contact_frame_transfer_oracle=a.contact_frame_transfer,
                  contact_frame_from_transfer_oracle=a.contact_frame_from_transfer,
                  contact_landing=a.contact_landing,
                  reset_contact_frame_each_cycle=a.reset_contact_frame_each_cycle,
                  unload_lookahead_seconds=a.lookahead_ms/1000.,lookahead_scope='Privileged exact scene/state shadow MuJoCo; left unload only; not a deployable perception-based guarantee',reference=str(a.plan.resolve()),
                  acceptance='Legacy checks plus native joint position bound at every physics step; early actual-contact controller with baseline contact establishment gate',
                  sampling='Deterministic grid, not a statistical success probability. X=world depth; Z=world height.',
                  force_source='Ideal MuJoCo contact forces; no sensor noise/delay.',
                  guard='Before removing a support, other three must carry >=10 N on expected rods at <=20 mm/s for 250 ms. On denial, hold prior reference using only confirmed previous contacts for up to 500 ms then end experiment. No retreat claim.')
    (a.out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
    results=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for result in pool.map(run_case,[(n,o,a.guard,a.out,str(a.plan.resolve()),a.failure_hold,a.lookahead_ms/1000.,a.relax_tracking,a.contact_frame,a.contact_frame_transfer,a.contact_frame_from_transfer,a.contact_landing,a.reset_contact_frame_each_cycle) for n,o in cases]):
            results.append(result)
            (a.out/'results.json').write_text(json.dumps(results,indent=2))
            print(json.dumps({k:result[k] for k in ('name','passed','reason','time_s')}),flush=True)
    print(f'Passed {sum(r["passed"] for r in results)}/{len(results)}',flush=True)


if __name__=='__main__':main()
