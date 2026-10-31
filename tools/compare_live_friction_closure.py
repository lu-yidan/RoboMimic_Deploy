"""Compare audited baseline and causal friction-estimate candidate, known case only."""
import json
from pathlib import Path
from tools.continue_vector_full_cycle_training import sha


def main():
    rows=[]
    for name in ('live_closure_trace_20261023_v1','measured_friction_closure_20261023_v1'):
        p=Path('outputs')/name/'gain0_drop1_foot_z_plus15'
        s=json.loads((p/'summary.json').read_text());a=json.loads((p/'independent_audit.json').read_text())
        live=json.loads((p/'live_trace_audit.json').read_text());v=json.loads((p/'velocity_response_audit.json').read_text())
        assert a['integrity_passed'] and live['passed']
        assert sha(p/'summary.json')==a['summary_sha256'] and sha(p/'trajectory.npz')==a['trajectory_sha256']
        assert sha(p/'live_trace.npz')==live['trace_sha256']==v['trace_sha256']
        rows.append(dict(experiment=name,accepted=a['independently_accepted'],time_s=s['runtime']['time_s'],
            joint_closure_mrad=s['endpoints'][0]['joint_closure_rad']*1000 if s['endpoints'][0]['joint_closure_rad'] is not None else None,
            root_rise_max_mm=s['root_rise_max_m']*1000 if s['root_rise_max_m'] is not None else None,
            live_sample_count=live['sample_count'],root_response_error_rms=v['root_response_error_rms'],
            final_contact_world_force=live['final_contact_world_force_mean'],audit_sha256=sha(p/'independent_audit.json'),
            velocity_audit_sha256=sha(p/'velocity_response_audit.json')))
    out=Path('outputs/live_friction_comparison_20261023_v1');out.mkdir(exist_ok=False)
    result=dict(rows=rows,baseline_replaced=False,new_training_steps=0,
        scope='One known +15mm case. The correction changes the simulator-oracle friction estimate only; no held-out robustness or hardware claim. Promotion requires original successful cases to remain successful.')
    (out/'summary.json').write_text(json.dumps(result,indent=2));(out/'comparison_source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
