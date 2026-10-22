"""Five known-height full cycles using the bounded support reference candidate."""
import json
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from tools.evaluate_reference_transfer import run_case
from tools.continue_vector_full_cycle_training import sha


def main():
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
    source=Path('outputs/support_retarget_plan_v1');audit=json.loads((source/'independent_audit.json').read_text())
    assert audit['passed'] and audit['plan_sha256']==sha(source/'plan.npz') and audit['manifest_sha256']==sha(source/'manifest.json')
    output=Path('outputs/support_retarget_calibration_v1');output.mkdir(exist_ok=False)
    jobs=[(str(output),f'true_height_{h:g}mm',str(source/'plan.npz'),-3.,f'foot_z_plus{h:g}',{'rung_01':[0.,0.,h/1000]}) for h in (12.5,15.,10.,5.,7.5)]
    protocol=dict(jobs=jobs,source_sha256=sha(__file__),plan_audit_sha256=sha(source/'independent_audit.json'),
        selection='Accept only if at least one strict failure repaired and no original success regresses. Same candidate in all five known heights.',
        scope='Temporary reference retargeting from an estimated10mm support offset. No scene-specific retuning, no policy training; all five cases known calibration cases.')
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2));(output/'runner_source.py').write_bytes(Path(__file__).read_bytes())
    results=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(run_case,j) for j in jobs]):
            r=future.result();results.append(r);(output/'results.json').write_text(json.dumps(results,indent=2));print(json.dumps(r),flush=True)


if __name__=='__main__':main()
