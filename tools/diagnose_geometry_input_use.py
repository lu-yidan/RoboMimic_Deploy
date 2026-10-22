"""Counterfactual actor input sensitivity on recorded states; no new physics."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from training.bar_geometry_observation import ROWS
from tools.continue_vector_full_cycle_training import sha


def main():
    parser=argparse.ArgumentParser();parser.add_argument('experiment',type=Path);args=parser.parse_args()
    p=args.experiment;m=json.loads((p/'manifest.json').read_text());s=json.loads((p/'summary.json').read_text())
    a=json.loads((p/'independent_training_audit.json').read_text())
    if m['geometry_mode']!='oracle' or not a['passed'] or s.get('evaluation_pending'):
        raise ValueError('Completed audited oracle experiment required')
    if sha(p/'summary.json')!=a['experiment_summary_sha256'] or sha(p/'policy.zip')!=a['checkpoint_sha256']:
        raise ValueError('Evidence changed')
    torch.set_num_threads(1);model=PPO.load(p/'policy.zip',device='cpu')
    rows=[]
    for row in s['after']:
        path=p/f"after_{row['case']}.npz"
        if sha(path)!=row['trajectory_sha256']:raise ValueError('Trajectory changed')
        with np.load(path) as z:
            differences=[]
            for observation,action in zip(z['observations'],z['actions']):
                assert np.array_equal(model.predict(observation,deterministic=True)[0],action)
                blind=observation.copy();blind[136:]=0.
                differences.append(float(abs(model.predict(blind,deterministic=True)[0]-action).max()))
            differences=np.asarray(differences)
            early=differences[z['time_s'][:-1]<40.]
            rows.append(dict(case=row['case'],success=row['success'],samples=len(differences),
                max_abs_normalized_action_change=float(differences.max()),mean_abs_max_action_change=float(differences.mean()),
                fraction_changed_above_1e_minus6=float(np.mean(differences>1e-6)),
                max_before40s=float(early.max()) if len(early) else None,trajectory_sha256=sha(path)))
    weight=model.policy.features_extractor.geometry_projection.weight.detach().cpu().numpy()
    result=dict(cases=rows,geometry_weight_column_norms={f'{row}_{axis}':float(np.linalg.norm(weight[:,3*i+j]))
        for i,row in enumerate(ROWS) for j,axis in enumerate('xyz')},
        checkpoint_sha256=sha(p/'policy.zip'),audit_sha256=sha(p/'independent_training_audit.json'),source_sha256=sha(__file__),
        scope='Zeroing geometry inputs at the same recorded states. Nonzero action changes show sensitivity, not improved closed-loop control or a validated sensor.')
    (p/'geometry_input_sensitivity.json').write_text(json.dumps(result,indent=2))
    (p/'geometry_input_sensitivity_source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
