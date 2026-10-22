"""Separate actual height mismatch from the declared contact-normal bias."""
import json
import os
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
from tools.evaluate_reference_transfer import run_case
from tools.continue_vector_full_cycle_training import sha


def main():
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
    output=Path('outputs/highrod_unbiased_height_v1');output.mkdir(exist_ok=False)
    plan='outputs/highrod_endpoint_adapt_plan_v1/plan.npz'
    jobs=[(str(output),f'true_height_{h:g}mm',plan,0.,f'foot_z_plus{h:g}',{'rung_01':[0.,0.,h/1000]}) for h in (12.5,15.,10.,5.,7.5)]
    protocol=dict(jobs=jobs,source_sha256=sha(__file__),reference_sha256=sha(plan),
        scope='Five known heights, zero actions, same reference/controller including contact memory. Only declared contact-normal bias changes from-3deg to0deg. No perception estimator or generalization claim.')
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2));(output/'runner_source.py').write_bytes(Path(__file__).read_bytes())
    results=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(run_case,j) for j in jobs]):
            r=future.result();results.append(r);(output/'results.json').write_text(json.dumps(results,indent=2));print(json.dumps(r),flush=True)


if __name__=='__main__':main()
