"""Freeze new geometry points and three already-selected controller identities."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import numpy as np
from tools.compare_audited_closure_candidates import sha, load_run


def main():
    p = argparse.ArgumentParser()
    for name in ('baseline', 'trained', 'refined', 'previous-protocol', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    baseline, trained = load_run(a.baseline), load_run(a.trained)
    entries = {}
    manifests = []
    for role in ('baseline', 'trained', 'refined'):
        path = getattr(a, role)
        manifest = json.loads((path/'manifest.json').read_text())
        manifests.append(manifest)
        entries[role] = dict(directory=str(path.resolve()), manifest_sha256=sha(path/'manifest.json'),
            checkpoint_sha256=sha(path/'policy.zip'), command=manifest['adaptive_reference_command'],
            reference_sha256=manifest['reference_sha256'], prefix_checkpoint_sha256=manifest['prefix_checkpoint_sha256'],
            minimum_development_successes=8 if role == 'refined' else 7,
            refinement_transfer_gate_required=role == 'refined')
    for key in ('reference_sha256', 'controller_modifier', 'adaptive_reference_scales', 'prefix_checkpoint_sha256'):
        assert all(m[key] == manifests[0][key] for m in manifests), key
    old = json.loads(a.previous_protocol.read_text())
    used = [np.asarray(value['rung_01']) for value in [*baseline['cases'].values(), *old['cases'].values()]]
    rng = np.random.default_rng(2026102602)
    cases, strata = {}, {}
    for group, edges in (('interior', np.linspace(.006, .014, 5)), ('boundary', np.linspace(.014, .0155, 5))):
        for i in range(4):
            offset = [float(rng.uniform(-.005, .005)), 0., float(rng.uniform(edges[i], edges[i+1]))]
            assert all(not np.allclose(offset, previous, atol=1e-12, rtol=0) for previous in used)
            key = f'fresh_{group}_{i:02d}'
            cases[key] = {'rung_01': offset}; strata[key] = group
            used.append(np.asarray(offset))
    result = dict(created_at=datetime.now().astimezone().isoformat(), random_seed=2026102602,
        candidates=entries, cases=cases, strata=strata, previous_protocol_sha256=sha(a.previous_protocol),
        controller_modifier=manifests[0]['controller_modifier'], reference_scales=manifests[0]['adaptive_reference_scales'],
        source_sha256=sha(__file__),
        scope='Eight new predeclared geometry points, four interior and four boundary. Three controller identities and '
              'constant commands fixed before any fresh rollout. The refined candidate must first pass all eight known '
              'development cases and cached-to-full transfer audit. No hyperparameter, parameter or checkpoint selection '
              'on these fresh outcomes; retain failures. Finite deterministic points, not a success-probability estimate.')
    a.output.mkdir(exist_ok=False)
    (a.output/'protocol.json').write_text(json.dumps(result, indent=2))
    (a.output/'protocol_source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(dict(protocol_sha256=sha(a.output/'protocol.json'), cases=cases)))


if __name__ == '__main__':
    main()
