"""Compare audited fixed calibrations, explicitly separating them from PPO learning."""
import argparse,csv,json
from pathlib import Path
from tools.compare_audited_closure_candidates import load_run,sha


def main():
    p=argparse.ArgumentParser()
    for name in ('before','after','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();before=load_run(a.before);after=load_run(a.after)
    for key in ('checkpoint_sha256','reference_sha256','cases','controller_modifier'):assert before[key]==after[key]
    manifests=[json.loads((d/'manifest.json').read_text()) for d in (a.before,a.after)]
    assert manifests[1]['policy_adapter']=='closure-action-bias-v2'
    assert manifests[0]['adaptive_reference_scales']==manifests[1]['adaptive_reference_scales']
    rows=[]
    for case in before['cases']:
        old,new=before['rows'][case],after['rows'][case]
        row=dict(case=case)
        for label,data in [('before',old),('after',new)]:row.update({label+'_'+k:v for k,v in data.items()})
        rows.append(row)
    result=dict(before=before,after=after,rows=rows,
        reference_commands=[m.get('adaptive_reference_command') for m in manifests],
        action_adapters=[dict(tag=m.get('policy_adapter'),spec=m.get('closure_action_bias')) for m in manifests],
        before_successes=sum(r['before_success'] for r in rows),after_successes=sum(r['after_success'] for r in rows),
        gained=[r['case'] for r in rows if r['after_success'] and not r['before_success']],
        lost=[r['case'] for r in rows if r['before_success'] and not r['after_success']],source_sha256=sha(__file__),
        scope='Same checkpoint and unchanged physical/action limits; explicitly different fixed joint/reference calibration. Complete audited cases. Not newly learned PPO weights, automatic promotion, or broad generalization.')
    a.output.mkdir(exist_ok=False);(a.output/'comparison.json').write_text(json.dumps(result,indent=2))
    with (a.output/'comparison.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (a.output/'source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({k:result[k] for k in ('before_successes','after_successes','gained','lost')}),flush=True)


if __name__=='__main__':main()
