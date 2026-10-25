"""Serial, reproducible comparisons without concurrent GPU timing interference."""
import argparse,json,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);a=p.parse_args()
root=Path(__file__).resolve().parents[1];cases=json.loads(a.manifest.read_text());out=a.manifest.parent
for case in cases:
 for label,mode,ref in [('direct3',2,'direct'),('direct5',4,'direct'),('gmr14',7,'baseline')]:
  dest=out/(case['name']+'_'+label)
  if (dest/'report.json').exists():continue
  print('RUN',dest.name,flush=True)
  with (out/(dest.name+'.log')).open('w') as log:
   subprocess.run([sys.executable,str(root/'tools/evaluate_scalebfm_motion.py'),'--motion',case[ref],'--out',str(dest),'--control-mode',str(mode),'--no-video'],cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
  r=json.loads((dest/'report.json').read_text());print('DONE',dest.name,r['completed'],r['completed_seconds'],flush=True)
