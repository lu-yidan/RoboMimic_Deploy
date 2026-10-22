"""Complete five-case evaluation of a frozen composite reference."""
import json
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from tools.evaluate_reference_transfer import run_case
from tools.continue_vector_full_cycle_training import sha


def main():
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
    output=Path('outputs/support_endpoint075_calibration_v1');output.mkdir(exist_ok=False)
    plan=Path('outputs/support_endpoint075_plan_v1/plan.npz')
    jobs=[(str(output),f'true_height_{h:g}mm',str(plan),-3.,f'foot_z_plus{h:g}',{'rung_01':[0.,0.,h/1000]}) for h in (15.,12.5,10.,5.,7.5)]
    protocol=dict(jobs=jobs,source_sha256=sha(__file__),plan_sha256=sha(plan),
        selection='Accept only if at least one support-only failure repaired and no support-only success regresses. Retain all five known cases.',
        scope='Same support reference plus reused endpoint gain0.75. Endpoint targets differ; do not merge into original policy success counts. Not new training or held-out evaluation.')
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2));(output/'runner_source.py').write_bytes(Path(__file__).read_bytes())
    results=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(run_case,j) for j in jobs]):
            r=future.result();results.append(r);(output/'results.json').write_text(json.dumps(results,indent=2));print(json.dumps(r),flush=True)


if __name__=='__main__':main()
