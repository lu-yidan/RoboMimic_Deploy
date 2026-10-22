"""Compare identical finite scenarios while distinguishing method assumptions."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text())
def key(r):return r['case'],float(r['normal_bias_y_deg'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--ppo',type=Path,nargs='+',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    base_manifest=read(a.ppo[0]/'manifest.json');expected={(case,float(angle)) for case in base_manifest['cases'] for angle in base_manifest['evaluation_angles_deg']}
    methods={};proof={}
    for i,run in enumerate(a.ppo):
        manifest=read(run/'manifest.json');summary=read(run/'summary.json')
        assert read(run/'independent_audit.json')['integrity_passed']
        if 'parent_run' in manifest:assert read(run/'continuation_audit.json')['passed']
        assert manifest['prefixes']==base_manifest['prefixes']
        for rel in ['training/contact_load_transfer_residual_env.py','training/contact_foot_residual_env.py','training/climb_cycle.py','training/wall_support_env.py']:
            assert manifest['sources'][rel]==base_manifest['sources'][rel]
        if i==0:methods['zero']={key(r):r for r in summary['zero']}
        methods[f"PPO_{summary['actual_steps']}"]={key(r):r for r in summary['after']}
        proof[str(run)]=hashlib.sha256((run/'summary.json').read_bytes()).hexdigest()
    assumptions={'zero':'Biased normals, exact contact points/topology, ideal forces'}
    for name,folder in [('constant','constant_residual_comparison_v1'),('memory_only','early_contact_memory_only_v1'),('interval_and_memory','early_contact_memory_comparison_v1')]:
        run=ROOT/'outputs'/folder;manifest=read(run/'manifest.json');rows=read(run/'results.json')
        assert manifest['prefixes']==base_manifest['prefixes']
        for rel in ['training/contact_load_transfer_residual_env.py','training/contact_foot_residual_env.py','training/climb_cycle.py','training/wall_support_env.py']:
            assert manifest['sources'][rel]==base_manifest['sources'][rel],(name,rel)
        assert all(r['integrity_passed'] for r in rows)
        methods[name]={key(r):r for r in rows};proof[str(run)]=hashlib.sha256((run/'results.json').read_bytes()).hexdigest()
        if 'memory' in name:assert read(run/'memory_bounds_audit.json')['passed']
    assumptions.update(constant='Same controller information as PPO; fixed action selected on one known failure, then checked across15cases',memory_only='Additional exact geometric gap and <=20ms support memory; no uncertainty-cone tightening',interval_and_memory='Additional known world-Y +/-3deg interval and exact geometric-gap memory')
    for name,rows in methods.items():
        assert set(rows)==expected and len(rows)==15,name
        if name.startswith('PPO_'):assumptions[name]='Same biased controller/actor information as zero; one training seed, finite episode distribution'
    counts={name:dict(successes=sum(r['success'] for r in rows.values()),count=len(rows),added_vs_zero=[list(k) for k in sorted(expected) if rows[k]['success'] and not methods['zero'][k]['success']],lost_vs_zero=[list(k) for k in sorted(expected) if not rows[k]['success'] and methods['zero'][k]['success']]) for name,rows in methods.items()}
    output=dict(counts=counts,assumptions=assumptions,source_hashes=proof,scope='Paired deterministic scenarios and matched prefixes/core. Not a probability, publication novelty proof, or claim of equal information for memory methods. Constant action selected using one of the15cases.',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    a.out.mkdir(parents=True,exist_ok=False);(a.out/'summary.json').write_text(json.dumps(output,indent=2));print(json.dumps(output,indent=2))


if __name__=='__main__':main()
