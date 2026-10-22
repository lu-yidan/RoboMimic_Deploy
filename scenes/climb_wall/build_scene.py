"""Build a continuous rung wall for a 2 m ascent; generate an IK inspection pose."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def vec(values):
    return ' '.join(f'{float(v):.12g}' for v in values)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def solve_arm(m, d, side, target, quat):
    joints = [j for j in range(m.njnt) if m.joint(j).name.startswith(side+'_')
              and any(part in m.joint(j).name for part in ('shoulder','elbow','wrist'))]
    assert len(joints) == 7
    qadr = m.jnt_qposadr[joints]; dofs = m.jnt_dofadr[joints]
    bid = m.body('hook' if side == 'left' else 'right_hook').id
    for iteration in range(400):
        mujoco.mj_forward(m,d)
        rotation = np.zeros(3)
        mujoco.mju_subQuat(rotation,quat,d.xquat[bid])
        error = np.r_[target-d.xpos[bid],rotation]
        if np.linalg.norm(error) < 1e-7:
            break
        jp = np.zeros((3,m.nv)); jr = np.zeros_like(jp)
        mujoco.mj_jacBody(m,d,jp,jr,bid)
        jac = np.vstack((jp,jr))[:,dofs]
        delta = jac.T@np.linalg.solve(jac@jac.T+1e-6*np.eye(6),error)
        d.qpos[qadr] += np.clip(delta,-.04,.04)
        d.qpos[qadr] = np.clip(d.qpos[qadr],m.jnt_range[joints,0],m.jnt_range[joints,1])
    if np.linalg.norm(error) >= 1e-5:
        raise RuntimeError(f'{side} arm IK did not converge: {error}')
    return dict(side=side,iterations=iteration,error=error.tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parameters',type=Path,default=Path(__file__).with_name('parameters.json'))
    parser.add_argument('--source',type=Path,default=ROOT/'outputs/g1_hook_v2_mount_v2')
    parser.add_argument('--out',type=Path,default=ROOT/'outputs/climb_wall_2m_v3')
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Use a new output directory to preserve the previous scene')
    p = json.loads(args.parameters.read_text())
    source_manifest = json.loads((args.source/'assembly_manifest.json').read_text())
    for rel, sha in source_manifest['files_sha256'].items():
        assert digest(args.source/rel) == sha, rel
    rise_rows = round(p['climb_rise_m']/p['row_pitch_m'])
    assert abs(rise_rows*p['row_pitch_m']-p['climb_rise_m']) < 1e-9
    assert p['initial_hand_row']+rise_rows < p['row_count']
    args.out.mkdir(parents=True)
    shutil.copytree(args.source/'meshes',args.out/'meshes')
    shutil.copy2(args.source/'LICENSE',args.out/'LICENSE')
    tree = ET.parse(args.source/'model.xml'); root = tree.getroot()
    root.set('model','G1 V2 hooks - continuous rung wall - 2 m ascent')
    world = root.find('worldbody')
    for child in list(world):
        if child.tag != 'body':
            world.remove(child)
    angle = np.deg2rad(p['wall_tilt_deg']); tangent = np.tan(angle)
    height = p['wall_height_m']; radius = p['bar_radius_m']
    front = lambda z: p['wall_front_x_at_ground_m']+tangent*z
    bar_x = lambda z: front(z)-p['bar_horizontal_standoff_m']
    quat = vec([np.cos(angle/2),0,np.sin(angle/2),0])
    def geom(name,kind,**kw):
        attrs = dict(name=name,type=kind,contype='1',conaffinity='1',
                     friction=vec(p['assumed_friction']),solref='.004 1',solimp='.95 .99 .001')
        attrs.update({k:vec(v) if isinstance(v,(list,tuple,np.ndarray)) else str(v) for k,v in kw.items()})
        return ET.SubElement(world,'geom',attrs)
    geom('floor','plane',size='4 4 .1',rgba='.22 .25 .29 1')
    half_thickness = p['wall_thickness_m']/2
    geom('wall','box',pos=[front(height/2)+half_thickness/np.cos(angle),0,height/2],
         quat=quat,size=[half_thickness,p['wall_width_m']/2,height/2/np.cos(angle)],rgba='.32 .40 .46 1')
    support_y = p['bar_half_span_m']-.02
    for side,y in [('left',support_y),('right',-support_y)]:
        geom(f'{side}_mount_rail','box',pos=[front(height/2)-.01*np.cos(angle),y,height/2+.01*np.sin(angle)],
             quat=quat,size=[.012,.035,height/2/np.cos(angle)],rgba='.13 .16 .19 1')
        for i,z in enumerate(np.arange(.20,height,.4)):
            geom(f'{side}_anchor_{i}','capsule',fromto=[front(z)-.04,y,z,front(z)+.035,y,z],size='.006',rgba='.65 .67 .69 1')
    rows = []
    for i in range(p['row_count']):
        z = p['first_bar_height_m']+i*p['row_pitch_m']; x = bar_x(z)
        color = '.92 .52 .15 1' if i in (p['initial_foot_row'],p['initial_hand_row']) else '.28 .33 .38 1'
        if i == p['initial_hand_row']+rise_rows:
            color = '.20 .72 .61 1'
        name = f'rung_{i:02d}'
        geom(name,'capsule',fromto=[x,-p['bar_half_span_m'],z,x,p['bar_half_span_m'],z],size=str(radius),rgba=color)
        ET.SubElement(world,'site',name=name+'_target',pos=vec([x,0,z]),size='.003',group='5',rgba='0 1 0 1')
        for side,y in [('left',support_y),('right',-support_y)]:
            geom(f'{name}_{side}_arm','capsule',fromto=[x,y,z,front(z)+.005,y,z],size='.012',rgba='.19 .23 .27 1')
            geom(f'{name}_{side}_brace','capsule',fromto=[x+.025,y,z,front(z-.14)-.005,y,z-.14],size='.01',rgba='.19 .23 .27 1')
        rows.append(dict(index=i,name=name,center_m=[x,0,z],axis=[0,1,0],radius_m=radius,
                         usable_y_m=[-support_y+.06,support_y-.06]))
    for side in ('left','right'):
        foot = root.find(f".//body[@name='{side}_ankle_roll_link']")
        for g in foot.findall('geom'):
            if g.get('class') == 'foot':
                foot.remove(g)
        # Flat sole proxy replaces four isolated corner spheres, so midsole rungs
        # cannot pass between collision points. Keep original foot mass/inertia.
        ET.SubElement(foot,'geom',name=side+'_sole_surface',type='box',
                      pos='.035 0 -.028',size='.09 .034 .007',group='3',mass='0',
                      contype='1',conaffinity='1',condim='3',priority='1',
                      friction=vec(p['assumed_friction']),solref='.004 1',solimp='.95 .99 .001',rgba='.2 .8 .8 .4')
    ET.SubElement(world,'light',pos='-2 -2 5',dir='1 .4 -1',directional='true',ambient='0 0 0',diffuse='.35 .35 .35',specular='.1 .1 .1',castshadow='false')
    ET.SubElement(world,'light',pos='-1 3 3',dir='1 -1 -.4',directional='true',ambient='0 0 0',diffuse='.15 .15 .15',specular='0 0 0',castshadow='false')
    visual = root.find('visual')
    visual.find('global').set('offwidth','1600');visual.find('global').set('offheight','1200')
    ET.SubElement(visual,'headlight',ambient='.18 .18 .18',diffuse='.35 .35 .35',specular='.1 .1 .1')
    ET.indent(root);tree.write(args.out/'model.xml',encoding='utf-8')
    m = mujoco.MjModel.from_xml_path(str(args.out/'model.xml'));d = mujoco.MjData(m)
    d.qpos[:] = np.load(args.source/'initial_qpos.npy')
    # The previous crouch intersects intermediate rungs. Start with modest knee
    # flexion and horizontal soles; this is a kinematic seed, not a controller.
    bend = p['initial_knee_bend_rad']
    for side in ('left','right'):
        for part, value in [('hip_pitch',-bend/2),('knee',bend),('ankle_pitch',-bend/2)]:
            joint = m.joint(side+'_'+part+'_joint').id
            d.qpos[m.jnt_qposadr[joint]] = value
    mujoco.mj_forward(m,d)
    initial_quats = {side:d.xquat[m.body('hook' if side=='left' else 'right_hook').id].copy() for side in ('left','right')}
    ankle = d.xpos[m.body('left_ankle_roll_link').id].copy()
    foot_z = rows[p['initial_foot_row']]['center_m'][2]
    d.qpos[0] += bar_x(foot_z)-ankle[0]-p['foot_support_local_x_m']
    d.qpos[2] += foot_z+radius+.035-ankle[2]-p['initial_sole_compression_m']
    hand_z = rows[p['initial_hand_row']]['center_m'][2]
    ik = [solve_arm(m,d,side,np.array([bar_x(hand_z),y,hand_z-p['hook_center_below_bar_m']]),initial_quats[side])
          for side,y in [('left',.2),('right',-.2)]]
    mujoco.mj_forward(m,d);np.save(args.out/'initial_qpos.npy',d.qpos)
    report = dict(parameters=p,rows=rows,ik=ik,climb_rise_m=p['climb_rise_m'],
                  initial_foot_row=p['initial_foot_row'],goal_foot_row=p['initial_foot_row']+rise_rows,
                  initial_hand_row=p['initial_hand_row'],goal_hand_row=p['initial_hand_row']+rise_rows,
                  nominal_base_translation_m=[tangent*p['climb_rise_m'],0,p['climb_rise_m']],
                  initial_root_position_m=d.qpos[:3].tolist(),
                  assumptions=['Rigid wall, rods and braces; no structural load validation.',
                               'Sole box is an approximate contact surface; foot inertial properties are unchanged.',
                               '2 m is target vertical ascent, not verified robot climbing performance.',
                               'Wrist adapter remains a visual placeholder; rigid hook attachment assumed.'],
                  builder_sha256=digest(__file__),source_manifest_sha256=digest(args.source/'assembly_manifest.json'),
                  files_sha256={str(f.relative_to(args.out)).replace('\\','/'):digest(f) for f in args.out.rglob('*') if f.is_file()})
    (args.out/'scene_manifest.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:report[k] for k in ('initial_root_position_m','ik','nominal_base_translation_m')},indent=2))


if __name__ == '__main__':
    main()
