"""Development-only hook probing intervention, preserving the original first cycle."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from tools import evaluate_v2_multicycle as evaluator
from tools.validate_swing_reference import read,write,sha,ROOT
from training.physics_telemetry import install_physics_telemetry
from training.probe_hook_loading import install_hook_admission,ORIGINAL_COMMAND_SHA256

CONFIG=dict(probe_force_n=15.,confirm_force_n=5.,confirm_seconds=.05,ramp_seconds=.5,timeout_seconds=1.5)


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--two-cycle-gate',type=Path);a=p.parse_args()
    run=a.run.resolve();out=a.output.resolve();m=read(run/'manifest.json');accepted=read(run/'v2_multicycle_audit.json')
    assert accepted['passed'] and accepted['manifest_sha256']==sha(run/'manifest.json')
    assert accepted['trajectory_sha256']==sha(run/'trajectory.npz')
    assert m['cycles'] in (2,8) and not m.get('contact_admission')
    if m['cycles']==8:
        assert a.two_cycle_gate is not None
        gm=read(a.two_cycle_gate/'manifest.json');ga=read(a.two_cycle_gate/'hook_admission_audit.json')
        assert ga['passed'] and gm['contact_admission']['parameters']==CONFIG
        assert ga['manifest_sha256']==sha(a.two_cycle_gate/'manifest.json')
    state=[];original_class=evaluator.PhysicalV2CycleEnv;original_write=evaluator.write;argv=sys.argv
    protocol=dict(parameters=CONFIG,start='Only after the first complete cycle',
        target='Baseline fixed-contact QP during newly requested left/right hook support',
        force_bound_frame='World-vertical z component of estimated contact forces, not actual surface-normal or impact force',
        original_qp_command_sha256=ORIGINAL_COMMAND_SHA256,baseline_run=str(run),baseline_audit_sha256=sha(run/'v2_multicycle_audit.json'),
        baseline_trajectory_sha256=sha(run/'trajectory.npz'),source_sha256=sha(__file__),
        status='Predeclared development diagnostic; not adopted',new_training_steps=0,
        scope='Cap estimated world-vertical new-hook support at 15 N total, require intended-rod measured normal and vertical force >5 N for 50 ms, then release the cap over 0.5 s. Stop on 1.5 s confirmation timeout. QP caps do not bound physical collision forces. Ideal simulated force feedback; no hardware claim.')
    class ProbeEnv(original_class):
        def reset(self,**kwargs):
            observation=super().reset(**kwargs)
            state.append(dict(admission=install_hook_admission(self,**CONFIG),telemetry=install_physics_telemetry(self)))
            return observation
    def with_protocol(path,value):
        if Path(path)==out/'manifest.json':
            original_write(out/'hook_admission_protocol.json',protocol)
            value=dict(value,contact_admission=protocol,hook_admission_protocol_sha256=sha(out/'hook_admission_protocol.json'),
                scope=value['scope']+' Experimental later-cycle hook admission changes the fixed-contact QP world-vertical force caps; this is not the adopted controller.')
        return original_write(path,value)
    evaluator.PhysicalV2CycleEnv=ProbeEnv;evaluator.write=with_protocol
    sys.argv=['evaluate_v2_multicycle','--candidate',m['parent_experiment'],'--plan',m['plan_path'],'--case',m['case'],'--output',str(out)]
    if a.two_cycle_gate:sys.argv+=['--two-cycle-gate',str(a.two_cycle_gate.resolve())]
    try:evaluator.main()
    finally:evaluator.PhysicalV2CycleEnv=original_class;evaluator.write=original_write;sys.argv=argv
    assert len(state)==1
    telemetry=state[0]['telemetry'];np.savez_compressed(out/'physics_telemetry.npz',**{k:v[:telemetry['count']] for k,v in telemetry['data'].items()})
    write(out/'hook_admission_trace.json',state[0]['admission']['trace'])
    write(out/'hook_admission_record.json',dict(telemetry_sha256=sha(out/'physics_telemetry.npz'),trace_sha256=sha(out/'hook_admission_trace.json'),
        trajectory_sha256=sha(out/'trajectory.npz'),physics_samples=telemetry['count'],groups=telemetry['groups'],protocol_sha256=sha(out/'hook_admission_protocol.json')))
    subprocess.run([sys.executable,'-m','tools.audit_v2_multicycle',str(out)],cwd=ROOT,check=True)
    subprocess.run([sys.executable,'-m','tools.audit_hook_admission',str(out)],cwd=ROOT,check=True)


if __name__=='__main__':main()
