"""Fresh two-cycle open-loop ablation, matched to the pre-observed T12 pilot."""
import argparse
from pathlib import Path
import subprocess
import sys
import numpy as np
from tools import evaluate_v2_multicycle as evaluator
from tools.validate_swing_reference import ROOT,read,write,sha
from training.physics_telemetry import install_physics_telemetry
from training.timed_hook_loading import install_timed_hook_loading,PILOT_DELAYS


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--matched-probe',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run=a.run.resolve();probe=a.matched_probe.resolve();out=a.output.resolve()
    m=read(run/'manifest.json');gate=read(run/'v2_multicycle_audit.json')
    pm=read(probe/'manifest.json');pa=read(probe/'hook_admission_audit_v2.json')
    assert gate['passed'] and gate['manifest_sha256']==sha(run/'manifest.json')
    assert gate['trajectory_sha256']==sha(run/'trajectory.npz')
    assert pa['passed'] and pa['manifest_sha256']==sha(probe/'manifest.json')
    assert pa['trajectory_sha256']==sha(probe/'trajectory.npz')
    assert m['cycles']==2 and m['case']=='training_12' and not m.get('contact_admission')
    assert all(m[k]==pm[k] for k in ('case','cycles','offsets','plan_sha256','checkpoint_sha256','parent_experiment'))
    assert pm['contact_admission']['parameters']['probe_force_n']==15.
    assert pm['contact_admission']['parameters']['ramp_seconds']==.5
    for event in pa['events']:
        assert abs(event['confirmed_s']-event['start_s']-PILOT_DELAYS[event['group']])<1e-8
    protocol=dict(status='Open-loop ablation; not adopted',delays_seconds=PILOT_DELAYS,
        probe_force_n=15.,ramp_seconds=.5,feedback_used=False,new_training_steps=0,
        baseline_run=str(run),baseline_audit_sha256=sha(run/'v2_multicycle_audit.json'),
        matched_probe=str(probe),matched_probe_audit_sha256=sha(probe/'hook_admission_audit_v2.json'),
        matched_probe_trajectory_sha256=sha(probe/'trajectory.npz'),source_sha256=sha(__file__),
        scope='Replay the nominal pilot delays (right 85 ms, left 90 ms), then the same estimated world-vertical loading ramp. No contact feedback or confirmation timeout is used. This nominal comparator alone cannot establish robustness under uncertain contact.')
    states=[];old_class=evaluator.PhysicalV2CycleEnv;old_write=evaluator.write;old_argv=sys.argv
    class TimedEnv(old_class):
        def reset(self,**kwargs):
            observation=super().reset(**kwargs)
            states.append(dict(loading=install_timed_hook_loading(self),telemetry=install_physics_telemetry(self)))
            return observation
    def annotate(path,value):
        if Path(path)==out/'manifest.json':
            old_write(out/'timed_loading_protocol.json',protocol)
            value=dict(value,timed_loading=protocol,timed_loading_protocol_sha256=sha(out/'timed_loading_protocol.json'),
                scope=value['scope']+' Later hooks use the declared OPEN-LOOP timed loading ablation; not the adopted controller.')
        return old_write(path,value)
    evaluator.PhysicalV2CycleEnv=TimedEnv;evaluator.write=annotate
    sys.argv=['evaluate_v2_multicycle','--candidate',m['parent_experiment'],'--plan',m['plan_path'],'--case',m['case'],'--output',str(out)]
    try:evaluator.main()
    finally:
        evaluator.PhysicalV2CycleEnv=old_class;evaluator.write=old_write;sys.argv=old_argv
        if states and (out/'manifest.json').exists():
            assert len(states)==1
            telemetry=states[0]['telemetry']
            np.savez_compressed(out/'physics_telemetry.npz',**{k:v[:telemetry['count']] for k,v in telemetry['data'].items()})
            write(out/'timed_loading_trace.json',states[0]['loading']['trace'])
            write(out/'timed_loading_record.json',dict(telemetry_sha256=sha(out/'physics_telemetry.npz'),
                trace_sha256=sha(out/'timed_loading_trace.json'),physics_samples=telemetry['count'],groups=telemetry['groups'],
                trajectory_sha256=sha(out/'trajectory.npz') if (out/'trajectory.npz').exists() else None,
                protocol_sha256=sha(out/'timed_loading_protocol.json')))
    subprocess.run([sys.executable,'-m','tools.audit_v2_multicycle',str(out)],cwd=ROOT,check=True)
    subprocess.run([sys.executable,'-m','tools.audit_timed_hook_loading',str(out)],cwd=ROOT,check=True)


if __name__=='__main__':main()
