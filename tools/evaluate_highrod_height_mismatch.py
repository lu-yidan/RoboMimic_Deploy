"""Fixed +10mm reference against actual height errors; no online retuning."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
import os
from pathlib import Path
from tools.evaluate_reference_transfer import run_case
from tools.continue_vector_full_cycle_training import sha


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    parent=Path('outputs/highrod_strict_reference_transfer_v1')
    audit=json.loads((parent/'adapted_bias-3/independent_audit.json').read_text())
    if not audit['independently_accepted'] or audit['summary_sha256']!=sha(parent/'adapted_bias-3/summary.json'):raise ValueError('Verified exact-height reference required')
    if not (parent/'paired_checks.json').exists():raise ValueError('Reference comparison must finish first')
    plan='outputs/highrod_endpoint_adapt_plan_v1/plan.npz'
    args.output.mkdir(exist_ok=False)
    heights=[5.,7.5,10.,12.5,15.]
    jobs=[(str(args.output),f'true_height_{h:g}mm',plan,-3.,f'foot_z_plus{h:g}',{'rung_01':[0.,0.,h/1000.]}) for h in heights]
    protocol=dict(jobs=jobs,assumed_height_mm=10.,true_heights_mm=heights,errors_true_minus_assumed_mm=[h-10. for h in heights],
        plan_sha256=sha(plan),runner_sha256=sha(__file__),rollout_sha256=sha('tools/evaluate_reference_transfer.py'),
        parent_audit_sha256=sha(parent/'adapted_bias-3/independent_audit.json'),
        parent_paired_checks_sha256=sha(parent/'paired_checks.json'),
        scope='Five fixed actual heights with one offline +10mm reference. Known selected reference; no sensor or estimator is implemented. Zero actions and strict thresholds fixed. This is true-height mismatch, not the previous correction-amplitude experiment. Separate task reference, not original seven-case policy evaluation.')
    (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2));(args.output/'runner_source.py').write_bytes(Path(__file__).read_bytes())
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
    results=[]
    with ProcessPoolExecutor(max_workers=3) as pool:
        for future in as_completed([pool.submit(run_case,j) for j in jobs]):
            r=future.result();results.append(r);(args.output/'results.json').write_text(json.dumps(results,indent=2));print(json.dumps(r),flush=True)


if __name__=='__main__':main()
