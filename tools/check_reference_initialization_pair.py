"""Verify the controlled initialization difference before feedback PPO training."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from tools.continue_vector_full_cycle_training import state_difference
from tools.compare_audited_closure_candidates import sha


def main():
    parser = argparse.ArgumentParser()
    for key in ('original', 'refined', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    paths = [args.original, args.refined]
    protocols, models = [], []
    for path in paths:
        summary = json.loads((path/'summary.json').read_text())
        assert summary['passed'] and summary['protocol_sha256'] == sha(path/'protocol.json')
        assert summary['checkpoint_sha256'] == sha(path/'policy_initial.zip')
        protocols.append(json.loads((path/'protocol.json').read_text()))
        models.append(PPO.load(path/'policy_initial.zip', device='cpu'))
    keys = ['teacher_sha256','controller_modifier','delta_scale','hold_steps','physical_gamma','config','cases','shards_summary_sha256','observation_contract']
    checks = {key: protocols[0][key] == protocols[1][key] for key in keys}
    checks.update(policy_parameters_exact=state_difference(models[0].policy.state_dict(),models[1].policy.state_dict())==0,
                  optimizer_exact=state_difference(models[0].policy.optimizer.state_dict(),models[1].policy.optimizer.state_dict())==0,
                  initial_steps_zero=models[0].num_timesteps==models[1].num_timesteps==0,
                  observation_low_exact=bool(np.array_equal(models[0].observation_space.low,models[1].observation_space.low)),
                  observation_high_exact=bool(np.array_equal(models[0].observation_space.high,models[1].observation_space.high)))
    parents = [json.loads((Path(protocol['parent'])/'manifest.json').read_text()) for protocol in protocols]
    for key in ('reference_sha256','bundle_manifest_sha256','adaptive_reference_scales','prefix_checkpoint_sha256'):
        checks[key] = parents[0][key] == parents[1][key]
    assert protocols[0]['seed_command'] != protocols[1]['seed_command']
    result = dict(passed=all(checks.values()), checks=checks,
        protocol_sha256=[sha(path/'protocol.json') for path in paths],
        checkpoint_sha256=[sha(path/'policy_initial.zip') for path in paths],
        commands=[protocol['seed_command'] for protocol in protocols], source_sha256=sha(__file__),
        scope='Same frozen teacher, actor/critic initialization, optimizer, training cases, reward interface and hyperparameters; constant reference seed differs. This is an initialization check, not a trained outcome or multi-seed statistical comparison.')
    args.output.mkdir(exist_ok=False)
    (args.output/'summary.json').write_text(json.dumps(result,indent=2))
    (args.output/'source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(result))
    assert result['passed']


if __name__ == '__main__':
    main()
