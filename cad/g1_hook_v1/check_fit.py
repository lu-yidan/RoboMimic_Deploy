"""Check geometric entry/crown contact of the concept hook; no dynamics claim."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import mujoco


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',type=Path,default=Path(__file__).parent/'model')
    a=parser.parse_args(); root=a.model
    report_path=root/'fit_check.json'
    if report_path.exists():
        raise FileExistsError('Preserve the prior geometric check')
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    for name,digest in manifest['files_sha256'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
    p=json.loads((root/'parameters.json').read_text(encoding='utf-8'))
    model=mujoco.MjModel.from_xml_path(str(root/'hook_fit.xml')); data=mujoco.MjData(model)
    assert model.nq==7 and model.neq==0
    np.testing.assert_allclose(model.body_mass[1],manifest['mass_kg'],rtol=1e-10)
    np.testing.assert_allclose(model.body_ipos[1],manifest['center_of_mass_m'],atol=1e-12)
    rotation=np.zeros(9);mujoco.mju_quat2Mat(rotation,model.body_iquat[1]);rotation=rotation.reshape(3,3)
    np.testing.assert_allclose(rotation@np.diag(model.body_inertia[1])@rotation.T,
                               manifest['inertia_about_com_kg_m2'],rtol=1e-6,atol=1e-12)
    seat_z=(p['bar_diameter']/2-p['inner_radius'])/1000
    rows=[]
    for z in np.r_[np.linspace(.10,seat_z+.0002,160),seat_z-.0002]:
        data.qpos[:]=[0,0,z,1,0,0,0]
        mujoco.mj_forward(model,data)
        rows.append(dict(hook_z_m=float(z),contacts=data.ncon,
            minimum_distance_m=min([float(c.dist) for c in data.contact],default=None)))
    assert all(row['contacts']==0 for row in rows[:-1])
    assert rows[-1]['contacts']>0
    report=dict(geometry_check_passed=True,opening_unobstructed=True,seat_contact_detected=True,
        mass_and_com_inertia_match=True,path_samples=len(rows),mujoco=mujoco.__version__,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        model_sha256=hashlib.sha256((root/'hook_fit.xml').read_bytes()).hexdigest(),
        scope='Prescribed centered vertical entry at nominal diameter; no mj_step, dynamic hanging, off-axis release, wrist assembly, or strength test',
        rows=rows)
    report_path.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))


if __name__=='__main__':
    main()
