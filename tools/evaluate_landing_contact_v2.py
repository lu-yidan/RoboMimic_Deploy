"""Full dynamics after nominal and estimated-geometry gates pass."""
import json
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from tools.evaluate_reference_transfer import run_case
from tools.continue_vector_full_cycle_training import sha


def main():
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
    out=Path('outputs/landing_contact_calibration_v2');plan=Path('outputs/landing_contact_plan_v4/plan.npz');manifest=json.loads((plan.parent/'manifest.json').read_text())
    assert manifest['passed'] and manifest['plan_sha256']==sha(plan)
    gates={}
    for name in ('nominal','estimated'):
        p=plan.parent/(name+'_audit.json');assert json.loads(p.read_text())['passed'];gates[str(p)]=sha(p)
    out.mkdir(exist_ok=False)
    jobs=[(str(out),f'true_height_{h:g}mm',str(plan),-3.,f'foot_z_plus{h:g}',{'rung_01':[0.,0.,h/1000]}) for h in (15.,12.5,10.,5.,7.5)]
    protocol=dict(jobs=jobs,plan_sha256=sha(plan),geometry_gates=gates,source_sha256=sha(__file__),selection='Repair at least one support-reference failure without regression on any support-reference success.',scope='New landing/reference family, including estimated height and closure tangency changes. No single-mechanism causal claim, no policy learning. Same original strict task limits; changed terminal target explicit.')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2));(out/'runner_source.py').write_bytes(Path(__file__).read_bytes());rows=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(run_case,j) for j in jobs]):
            r=future.result();rows.append(r);(out/'results.json').write_text(json.dumps(rows,indent=2));print(json.dumps(r),flush=True)


if __name__=='__main__':main()
