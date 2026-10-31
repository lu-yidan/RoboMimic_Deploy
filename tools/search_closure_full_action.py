"""Two bounded CEM generations in the existing 26D action space.

Each candidate is a smooth fixed feedforward over a real 14 s suffix. This is
known-case teacher search, not policy-gradient training or complete-cycle proof.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import numpy as np
from tools.probe_closure_action_timing import run
from tools.continue_vector_full_cycle_training import sha


def main():
    seed=Path('outputs/closure_seed_policy_20261024_v1')
    gate=json.loads((seed/'summary.json').read_text());assert gate['passed']
    center=np.asarray(gate['seed_bias']);duration=gate['seed_ramp_seconds']
    out=Path('outputs/closure_full_action_search_20261024_v1');out.mkdir(exist_ok=False)
    protocol=dict(seed=20261024,generations=2,population=32,elites=8,
      initial_bias=center.tolist(),ramp_seconds=duration,initial_std_interior=.12,initial_std_boundary=.02,
      action_bias_bounds=[-1.,1.],early_stop_passing_cost=.02,
      seed_gate_sha256=sha(seed/'summary.json'),parent_checkpoint_sha256=gate['parent_checkpoint_sha256'],
      cache_summary_sha256=sha('outputs/closure_prefix_cache_20261024_v4/summary.json'),
      sources={name:sha(name) for name in [str(Path(__file__).resolve()),'tools/probe_closure_action_timing.py','training/closure_curriculum_env_v3.py']},
      scope='At most 64 full-26D bounded bias candidates, two declared CEM generations. Fixed 1 s ramp, no per-step random noise. Known-case suffix search with extra simulator samples; full-cycle and grid gates remain mandatory.')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    rng=np.random.default_rng(protocol['seed']);std=np.where(abs(center)>.95,.02,.12)
    rows=[];history=[];best=None
    with ProcessPoolExecutor(max_workers=3) as pool:
        for generation in range(2):
            population=[center.copy() if best is None else np.asarray(best['parameters'])]
            population.extend(np.clip(center+rng.normal(size=(31,26))*std,-1.,1.))
            futures={};generation_rows=[]
            for bias in population:
                previous=next((r for r in rows if np.array_equal(np.asarray(r['parameters']),bias)),None)
                if previous is not None:
                    generation_rows.append(previous);continue
                trial=len(rows)+len(futures);directory=out/f'trial_{trial:03d}';directory.mkdir(exist_ok=False)
                future=pool.submit(run,(duration,bias,bias.tolist(),str(directory.resolve())))
                futures[future]=(trial,directory)
            for future in as_completed(futures):
                row=future.result();trial,directory=futures[future]
                row.update(trial=trial,generation=generation,directory=str(directory.resolve()),parameters_are='26D normalized action bias')
                (directory/'candidate.json').write_text(json.dumps(row,indent=2))
                rows.append(row);generation_rows.append(row)
                print(json.dumps(dict(generation=generation,trial=trial,success=row['success'],cost=row['cost'],**row['metrics'])),flush=True)
            ranked=sorted(generation_rows,key=lambda r:(not r['success'],r['cost']))
            best=min(rows,key=lambda r:(not r['success'],r['cost']))
            history.append(dict(generation=generation,best_trial=best['trial'],best_success=best['success'],best_cost=best['cost'],successes=sum(r['success'] for r in generation_rows)))
            (out/'progress.json').write_text(json.dumps(dict(history=history,best=best,completed_candidates=len(rows)),indent=2))
            if best['success'] and best['cost']<=.02:break
            elite=np.asarray([r['parameters'] for r in ranked[:8]])
            center=.7*elite.mean(axis=0)+.3*np.asarray(best['parameters'])
            std=np.clip(elite.std(axis=0),.015,.2)
    rows.sort(key=lambda r:r['trial'])
    result=dict(completed=True,rows=rows,history=history,best=best,basis=np.eye(26).tolist(),
      groups=[f'action_{j:02d}' for j in range(26)],protocol_sha256=sha(out/'protocol.json'),scope=protocol['scope'])
    (out/'summary.json').write_text(json.dumps(result,indent=2))
    (out/'search_source.py').write_bytes(Path(__file__).read_bytes())
    (out/'rollout_source.py').write_bytes(Path('tools/probe_closure_action_timing.py').read_bytes())
    print(json.dumps(dict(completed=len(rows),best_trial=best['trial'],success=best['success'],cost=best['cost'])),flush=True)


if __name__=='__main__':main()
