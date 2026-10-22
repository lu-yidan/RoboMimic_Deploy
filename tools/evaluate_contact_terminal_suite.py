"""Two predeclared terminal references, each evaluated on all five known heights."""
import json
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from tools.evaluate_reference_transfer import run_case
from tools.continue_vector_full_cycle_training import sha


def main():
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
    root=Path('outputs/support_contact_terminal_calibration_v1');root.mkdir(exist_ok=False);jobs=[];results={};hashes={}
    for target in (10.,15.):
        output=root/f'target{target:g}';output.mkdir();plan=Path(f'outputs/support_contact{target:g}_plan_v1/plan.npz')
        subset=[(str(output),f'true_height_{h:g}mm',str(plan),-3.,f'foot_z_plus{h:g}',{'rung_01':[0.,0.,h/1000]}) for h in (15.,12.5,10.,5.,7.5)]
        protocol=dict(jobs=subset,source_sha256=sha(__file__),plan_sha256=sha(plan),
            selection='Qualify only if at least one support-only failure repaired and no support-only success regresses. If both references qualify prefer estimated10mm, matching the original estimate.',
            scope='Known-case reference calibration, not sensing or policy training. Endpoint target changes; strict physical/closure limits remain fixed.')
        (output/'protocol.json').write_text(json.dumps(protocol,indent=2));(output/'runner_source.py').write_bytes(Path(__file__).read_bytes())
        jobs.extend(subset);results[str(output)]=[];hashes[str(output/'protocol.json')]=sha(output/'protocol.json')
    (root/'protocol.json').write_text(json.dumps(dict(subprotocol_hashes=hashes,source_sha256=sha(__file__)),indent=2))
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures={pool.submit(run_case,job):job[0] for job in jobs}
        for future in as_completed(futures):
            output=futures[future];r=future.result();results[output].append(r);(Path(output)/'results.json').write_text(json.dumps(results[output],indent=2));print(json.dumps(dict(output=output,result=r)),flush=True)


if __name__=='__main__':main()
