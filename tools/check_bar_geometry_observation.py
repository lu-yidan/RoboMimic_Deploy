"""Actual-physics short gate; not full-cycle acceptance or training evidence."""
import json
from pathlib import Path
import numpy as np
from training.strict_full_cycle_residual_env import FullCycleResidualEnv
from training.bar_geometry_observation import BarGeometryObservation, ROWS
from deploy_mujoco.run_trained_full_cycle import sha


def main():
    parent=Path('outputs/strict_full_cycle_vector_continue69632_v1')
    manifest=json.loads((parent/'manifest.json').read_text())
    out=Path('outputs/bar_geometry_observation_gate_v1')
    out.mkdir(exist_ok=False)
    records=[]
    for case, offsets in manifest['evaluation_cases'].items():
        envs=[FullCycleResidualEnv(plan=manifest['reference'],cases={case:offsets}) for _ in range(3)]
        plain, blind, oracle=envs[0],BarGeometryObservation(envs[1],mode='blind'),BarGeometryObservation(envs[2],mode='oracle')
        wrappers=[plain,blind,oracle]
        try:
            obs=[e.reset(seed=19,options={'case':case})[0] for e in wrappers]
            expected=np.array([offsets.get(row,[0.,0.,0.]) for row in ROWS]).reshape(-1)/.02
            np.testing.assert_allclose(obs[2][136:],expected,rtol=0,atol=1e-7)
            assert np.array_equal(obs[1][136:],np.zeros(12))
            for _ in range(50):
                assert np.array_equal(obs[0],obs[1][:136]) and np.array_equal(obs[0],obs[2][:136])
                transitions=[e.step(np.zeros(26)) for e in wrappers]
                obs=[t[0] for t in transitions]
                for k in (1,2):
                    assert transitions[0][1:]==transitions[k][1:]
                    for field in ('qpos','qvel','ctrl'):
                        assert np.array_equal(getattr(envs[0].env.data,field),getattr(envs[k].env.data,field))
                assert not transitions[0][2] and not transitions[0][3]
            records.append(dict(case=case,simulation_time_s=float(plain.env.data.time),
                exact_original_observations_rewards_states_controls=True,oracle_normalized_offsets=obs[2][136:].tolist()))
        finally:
            for e in wrappers:e.close()
    result=dict(passed=True,cases=records,observation_dimensions=148,
        reference_sha256=sha(manifest['reference']),parent_manifest_sha256=sha(parent/'manifest.json'),
        sources={p:sha(p) for p in [__file__,'training/bar_geometry_observation.py','training/strict_full_cycle_residual_env.py']},
        scope='Seven-case1s physics/input smoke gate only. Oracle reads simulation geometry; no perception sensor, no new policy training, no full-cycle gate.')
    (out/'summary.json').write_text(json.dumps(result,indent=2))
    for p in result['sources']:(out/Path(p).name).write_bytes(Path(p).read_bytes())
    print(json.dumps(dict(passed=True,cases=len(records),dimensions=148)))


if __name__=='__main__':main()
