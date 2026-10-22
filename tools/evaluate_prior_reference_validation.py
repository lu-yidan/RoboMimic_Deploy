"""Freeze an interpolation and horizontal-offset validation grid before execution."""
import json
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from tools.evaluate_reference_transfer import run_case
from tools.continue_vector_full_cycle_training import sha


def main():
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
    out=Path('outputs/prior_reference_validation_v1');out.mkdir(exist_ok=False)
    plan='outputs/highrod_endpoint_adapt_plan_v1/plan.npz';selection=Path('outputs/support_retarget_calibration_v1/selection.json')
    assert json.loads(selection.read_text())['selected']
    jobs=[]
    for x in (0.,-.005,.005):
        for height in (.00625,.00875,.01125,.01375):
            case=f'foot_x{x*1000:+g}_z{height*1000:g}'
            jobs.append((str(out),case,plan,-3.,case,{'rung_01':[x,0.,height]}))
    protocol=dict(paired_grid_sha256=sha(Path("outputs/support_reference_validation_v1/protocol.json")),jobs=jobs,plan_sha256=sha(plan),selection_sha256=sha(selection),source_sha256=sha(__file__),acceptance='Original strict full-cycle runtime and saved trajectory audit; report every case and separate x groups.',scope='Retrospective paired evaluation of the pre-existing unmodified reference on the exact already-frozen twelve-case validation grid. No additional candidate selection or tuning. Comparator execution follows candidate results, so not an independently preregistered two-arm experiment.')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2));(out/'runner_source.py').write_bytes(Path(__file__).read_bytes());rows=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(run_case,job) for job in jobs]):
            r=future.result();rows.append(r);(out/'results.json').write_text(json.dumps(rows,indent=2));print(json.dumps(r),flush=True)


if __name__=='__main__':main()
