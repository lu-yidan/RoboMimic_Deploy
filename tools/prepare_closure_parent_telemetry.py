"""Prepare one verified parent trajectory for non-invasive telemetry replay."""
import json,shutil
from pathlib import Path
from tools.continue_vector_full_cycle_training import sha


def main():
    source=Path('outputs/closure_margin_control98304_v1')
    candidate=Path('outputs/closure_waist_body_fullcycle_20261024_v1')
    out=Path('outputs/closure_parent_telemetry_replay_20261024_v1');out.mkdir(exist_ok=False)
    audit=json.loads((source/'independent_training_audit.json').read_text());assert audit['passed'] and audit['checkpoint_sha256']==sha(source/'policy.zip')
    m=json.loads((candidate/'manifest.json').read_text());old=json.loads((source/'manifest.json').read_text())
    case='foot_z_plus15';row=next(r for r in json.loads((source/'summary.json').read_text())['after'] if r['case']==case)
    assert sha(source/f'after_{case}.npz')==row['trajectory_sha256']
    for key in ('reference_sha256','normal_bias_y_deg','force_history_memory','bundle_manifest_sha256','geometry_mode','contract'):assert m[key]==old[key]
    m.update(evaluation_cases={case:old['evaluation_cases'][case]},training_cases={},prefix_checkpoint=str((out/'prefix_policy.zip').resolve()),
        controller_modifier=dict(waist_bias=[0.,0.,0.],body_drop_m=0.,waist_ramp_seconds=1.,geometry_gain='clip((raw_obs[138]-.625)/.125,0,1)'),
        scope='Original selected actor in both composite branches; zero additional waist/body command. Verified original failed highest-case trajectory reused only as exact replay target.')
    for key in ('seed_gate_sha256','seed_bias','seed_ramp_seconds','selection_summary_sha256'):m.pop(key,None)
    shutil.copytree(candidate/'source_snapshot',out/'source_snapshot')
    for name in ('policy.zip','prefix_policy.zip'):shutil.copy2(source/'policy.zip',out/name)
    shutil.copy2(source/f'after_{case}.npz',out/f'after_{case}.npz')
    (out/'manifest.json').write_text(json.dumps(m,indent=2))
    (out/'summary.json').write_text(json.dumps(dict(checkpoint_sha256=sha(out/'policy.zip'),after=[row],evaluation_pending=False,reused_parent_audit_sha256=sha(source/'independent_training_audit.json')),indent=2))
    (out/'after.json').write_text(json.dumps([row],indent=2))

if __name__=='__main__':main()
