"""Reproducible 15-case GMR-v10/ScaleBFM benchmark; preserves existing outputs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
from prepare_duobox_scalebfm import convert
ROOT=Path(__file__).resolve().parents[1]
CASES=[(0,'ma','jab'),(2,'ma','rear_uppercut'),(3,'ma','cross'),
       (6,'ma','lead_hook'),(7,'ma','rear_hook'),(4,'liu','jab_dodge'),
       (6,'liu','hook_dodge'),(7,'liu','rear_hook_dodge'),(0,'liu','guard'),
       (17,'ma','jab_cross'),(17,'liu','combo_defend'),(21,'ma','circle'),
       (22,'ma','pivot'),(27,'ma','moving_combo'),(27,'liu','moving_combo')]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--gmr-root',type=Path,default=Path('/home/d080/workspace/GMR'))
    p.add_argument('--scale-root',type=Path,default=Path('/home/d080/workspace/ScaleBFM'))
    p.add_argument('--data-root',type=Path,default=Path('/home/d080/workspace/ready_to_react/data/DuoBox/fbx_export_joints'))
    a=p.parse_args();a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=False)
    (a.out/'motions').mkdir();(a.out/'tests').mkdir();(a.out/'logs').mkdir()
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    def prepare(index):
        cache=a.gmr_root/'motion_data/duobox_web/motions'/f'{index:03d}_geometry_v10'
        if index==27:
            cache=ROOT/'outputs/duobox_scalebfm/retarget_027_v10'
        if not (cache/'pair_motion.npz').exists():
            cache=a.out/'retarget'/f'{index:03d}_v10'
            source=next(a.data_root.glob(f'{index:03d}_*'))
            with (a.out/'logs'/f'retarget_{index:03d}.log').open('w') as log:
                subprocess.run([str(a.gmr_root/'.venv/bin/python'),str(a.gmr_root/'scripts/duobox_to_g1_pair.py'),
                                '--input',str(source),'--output',str(cache),'--body-anchor','--no-video'],
                               cwd=a.gmr_root,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        return cache/'pair_motion.npz'
    summary=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures={i:pool.submit(prepare,i) for i in sorted({c[0] for c in CASES})}
        for number,(index,actor,label) in enumerate(CASES,1):
            name=f'{index:03d}_{actor}_{label}'
            try:
                pair=futures[index].result()
                motion=a.out/'motions'/f'{name}.npz'
                convert(pair,actor,motion,a.gmr_root,a.scale_root)
                result=a.out/'tests'/name
                with (a.out/'logs'/f'{name}.log').open('w') as log:
                    subprocess.run([sys.executable,str(ROOT/'tools/evaluate_scalebfm_motion.py'),
                                    '--motion',str(motion),'--out',str(result),'--no-video'],
                                   cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
                report=json.loads((result/'report.json').read_text())
                row=dict(case=name,report=report)
                print(f'[{number}/15] {name}: {report["completed_seconds"]:.2f}/{report["requested_seconds"]:.2f}s; failure={report["failure"]}',flush=True)
            except Exception as exc:
                row=dict(case=name,error=str(exc));print(f'[{number}/15] {name}: ERROR {exc}',flush=True)
            summary.append(row)
            (a.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    if any('error' in row for row in summary):
        raise SystemExit(1)


if __name__=='__main__':main()
