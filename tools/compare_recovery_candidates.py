"""Paired flat-ground deployment evaluation; no robot/DDS connections.
Each policy receives exactly the same poses. Records traces for re-fall analysis.
"""
import argparse, json, os, subprocess, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
PROFILES=['ft_r0_9500','ft_r1_9999','ft_r2_9000','d4_9999']


def summarize(out,profiles=PROFILES):
    comparisons={}
    for profile in profiles:
        folder=out/profile
        summary=json.loads((folder/'summary.json').read_text())
        cases=json.loads((folder/'cases.json').read_text())
        refalls=0;after_hold=0;late_travel=[];tau=[];upright_times=[]
        for case in cases:
            r=json.loads((folder/(case['name']+'.json')).read_text())
            t=np.load(folder/(case['name']+'.npz'))
            nq=int(t['nq']);nv=int(t['nv']);s=t['state'];z=s[:,nq+nv];u=s[:,nq+nv+1];hold=s[:,nq+nv+2]
            reached=np.maximum.accumulate((z>=1.15)&(u>=.93))
            held=np.maximum.accumulate(hold>=1-1e-6)
            fallen=(z<.65)|(u<.5)
            # Require 0.2 seconds continuously to avoid counting a single noisy frame.
            def persistent(mask):return bool(np.any(np.convolve(mask.astype(int),np.ones(10,dtype=int),'valid')>=10))
            refalls+=persistent(fallen&reached);after_hold+=persistent(fallen&held)
            late_travel.append(float(np.linalg.norm(np.diff(s[-500:,:2],axis=0),axis=1).sum()))
            tau.append(max(r['peak_tau']))
            if r['first_upright_s']>=0:upright_times.append(r['first_upright_s'])
        summary.update({'refall_after_first_upright':refalls,'refall_after_stable_1s':after_hold,
                        'last10s_base_xy_path_mean_m':float(np.mean(late_travel)),
                        'peak_tau_p95':float(np.percentile(tau,95)),
                        'first_upright_median_s_reached_only':float(np.median(upright_times)) if upright_times else None})
        comparisons[profile]=summary
    (out/'comparison.json').write_text(json.dumps(comparisons,indent=2)+'\n')
    print(json.dumps(comparisons,indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument('--bank',type=Path,required=True)
    p.add_argument('--profiles',nargs='+',default=PROFILES);p.add_argument('--out',type=Path,required=True);p.add_argument('--per-direction',type=int,default=16)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--summarize-only',action='store_true');a=p.parse_args();profiles=a.profiles
    a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=True)
    if not a.summarize_only:
        bank=np.load(a.bank);cases=[]
        for label in ['supine','prone','left_side_down','right_side_down']:
            ids=np.flatnonzero(bank['labels']==label)[:a.per_direction]
            assert len(ids)==a.per_direction
            for j,k in enumerate(ids):cases.append({'name':f'{label}_{j:02d}','direction':label,'qpos':bank['qpos'][k].tolist(),'record':True})
        cases_path=a.out/'paired_cases.json';cases_path.write_text(json.dumps(cases))
        for profile in profiles:
            with (a.out/(profile+'.log')).open('w') as log:
                subprocess.run([sys.executable,str(ROOT/'tools/validate_ft12k_sim.py'),'--profile',profile,'--out',str(a.out/profile),'--cases',str(cases_path),'--workers',str(a.workers)],stdout=log,stderr=subprocess.STDOUT,check=True,env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
    summarize(a.out,profiles)

if __name__=='__main__':main()
