"""Locate recorded swing contact peaks before changing trajectory clearance."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    contacts=json.loads((a.run/'contact_log.json').read_text());result=[]
    for phase,group in [('rf_swing','right_foot'),('lf_swing','left_foot')]:
        events=[]
        for row in contacts:
            if row['phase']!=phase:continue
            for c in row['contacts']:
                if c['group']!=group:continue
                force=float(np.linalg.norm(c['force_on_robot_world_n']))
                if force>1.:events.append(dict(time_s=row['time_s'],force_n=force,robot_geom=c['robot_geom'],external_geom=c['external_geom'],distance_m=c['distance_m']))
        result.append(dict(phase=phase,samples_above1n=len(events),first=events[0] if events else None,peak=max(events,key=lambda r:r['force_n']) if events else None,last=events[-1] if events else None))
    output=dict(results=result,scope='50Hz contact-log samples, so maxima can be below runtime2000Hz peaks.',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (a.run/'swing_contact_diagnostic.json').write_text(json.dumps(output,indent=2));print(json.dumps(output,indent=2))


if __name__=='__main__':main()
