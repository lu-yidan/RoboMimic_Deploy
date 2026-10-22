"""Create a separate G1 V2 hook assembly scene; no controller or training claim."""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import shutil
import xml.etree.ElementTree as ET
import numpy as np
import trimesh

ROOT=Path(__file__).resolve().parents[2]
CAD=Path(__file__).resolve().parent/'model_final'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def vec(v):return ' '.join(f'{float(x):.12g}' for x in v)


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=ROOT/'outputs/g1_hook_v2_mount_v1')
    a=p.parse_args();source=ROOT/'assets/climb_mounted'
    if a.out.exists():raise FileExistsError('Preserve earlier assembly output')
    cm=json.loads((CAD/'manifest.json').read_text());sm=json.loads((source/'manifest.json').read_text())
    for rel,digest in cm['files_sha256'].items():assert sha(CAD/rel)==digest,rel
    for rel,digest in sm['output_sha256'].items():assert sha(source/rel)==digest,rel
    a.out.mkdir(parents=True);shutil.copytree(source/'meshes',a.out/'meshes')
    shutil.copytree(CAD/'parts',a.out/'meshes/v2_parts')
    shutil.copy2(source/'initial_qpos.npy',a.out/'initial_qpos.npy')
    shutil.copy2(source/'LICENSE',a.out/'LICENSE')
    parts=[]
    for row in cm['parts']:
        mesh=trimesh.load_mesh(CAD/'parts'/(row['name']+'.stl'),process=True)
        mass=mesh.volume*1e-9*row['assumed_density_kg_m3']
        parts.append((mass,mesh.center_mass*.001,mesh.moment_inertia*1e-15*row['assumed_density_kg_m3']))
    mass=sum(row[0] for row in parts)
    com=sum(m*c for m,c,_ in parts)/mass
    inertia=sum(I+m*(np.eye(3)*np.dot(c-com,c-com)-np.outer(c-com,c-com)) for m,c,I in parts)
    assert np.linalg.eigvalsh(inertia).min()>0
    tree=ET.parse(source/'model.xml');root=tree.getroot();root.set('model','G1 black modular hooks V2 - assembly inspection')
    assets=root.find('asset');fixture=ET.parse(CAD/'hook_fit.xml').getroot()
    for mesh in fixture.find('asset').findall('mesh'):
        asset=copy.deepcopy(mesh);asset.set('name','v2_'+asset.get('name'))
        if asset.get('file'):asset.set('file','v2_parts/'+asset.get('file'))
        assets.append(asset)
    source_hook=fixture.find("worldbody/body[@name='hook']")
    # Preserve nominal arm posture and leave about 0.2 mm crown/bar clearance.
    offset=np.array([0.,0.,.0149]);angle=np.deg2rad(60)
    rotation=np.array([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]])
    mounting=[]
    for side,name,site in [('left','hook','contact'),('right','right_hook','right_contact')]:
        body=root.find(f".//body[@name='{name}']")
        oldpos=np.fromstring(body.get('pos'),sep=' ')
        body.set('pos',vec(oldpos+rotation@offset))
        for child in list(body):body.remove(child)
        full=[inertia[0,0],inertia[1,1],inertia[2,2],inertia[0,1],inertia[0,2],inertia[1,2]]
        ET.SubElement(body,'inertial',mass=str(mass),pos=vec(com),fullinertia=vec(full))
        for geom in source_hook.findall('geom'):
            g=copy.deepcopy(geom);g.set('name',side+'_v2_'+g.get('name'));g.set('mass','0')
            if g.get('mesh'):g.set('mesh','v2_'+g.get('mesh'))
            if g.get('group')=='3':
                g.set('solref','.004 1');g.set('solimp','.95 .99 .001');g.set('friction','.6 .01 .001')
            body.append(g)
        ET.SubElement(body,'site',name=site,pos='0 0 .022',size='.002',rgba='.8 .6 .2 1')
        ET.SubElement(body,'site',name=side+'_wrist_sensor_site',pos='0 0 0',size='.001')
        # Amber display envelope only: intended adapter region, NOT a completed
        # mechanical connection. No mass or collision is silently assigned.
        ET.SubElement(body,'geom',name=side+'_adapter_envelope',type='box',
            pos='-.039 0 -.046',size='.009 .020 .021',contype='0',conaffinity='0',mass='0',
            group='4',rgba='.95 .58 .14 .28')
        mounting.append(dict(side=side,hook_body=name,old_mount_pos_m=oldpos.tolist(),
            new_mount_pos_m=(oldpos+rotation@offset).tolist(),local_crown_offset_m=offset.tolist()))
    # The complete assembly is modelled rigidly. Relative compliance, bolt
    # preload and liner retention are not present in this preliminary model.
    ET.indent(root);tree.write(a.out/'model.xml',encoding='utf-8')
    report=dict(task='static G1 V2 assembly inspection',hook_mass_kg=mass,hook_com_m=com.tolist(),
        hook_inertia_com_kg_m2=inertia.tolist(),mounting=mounting,
        geometry_source_sha256=sha(CAD/'manifest.json'),robot_source_sha256=sha(source/'manifest.json'),
        builder_sha256=sha(__file__),physical_wrist_adapter_verified=False,
        adapter_envelope='amber visual-only box; no physical adapter mass or constraints',
        rigid_assembly_assumption=True,old_policy_validated_on_new_hooks=False,
        expected_dof=dict(nq=36,nv=35,nu=29),files_sha256={str(f.relative_to(a.out)).replace('\\','/'):sha(f) for f in a.out.rglob('*') if f.is_file()})
    (a.out/'assembly_manifest.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:report[k] for k in ('task','hook_mass_kg','mounting')},indent=2))


if __name__=='__main__':main()
