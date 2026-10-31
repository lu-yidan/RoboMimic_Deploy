"""Inspect saved failure geometry beside independently verified solver telemetry."""
import argparse
import json
from pathlib import Path
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from training.full_cycle_residual_env import FullCycleResidualEnv
from tools.compare_audited_closure_candidates import load_run, sha


def main():
    p = argparse.ArgumentParser()
    for key in ('experiment', 'telemetry', 'output'):
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--case', default='fresh_boundary_03')
    a = p.parse_args()
    verified = load_run(a.experiment)
    assert not verified['rows'][a.case]['success']
    m = json.loads((a.experiment/'manifest.json').read_text())
    audit = json.loads((a.telemetry/'failure_telemetry_audit.json').read_text())
    assert audit['passed'] and audit['summary_sha256'] == sha(a.telemetry/'summary.json')
    metric = next(row for row in audit['rows'] if row['case'] == a.case)
    a.output.mkdir(exist_ok=False)
    env = FullCycleResidualEnv(plan=m['reference'], cases={a.case:m['evaluation_cases'][a.case]})
    assert sha(env.env.bundle/'manifest.json') == m['bundle_manifest_sha256']
    env.reset(options={'case':a.case})
    e = env.env
    with np.load(a.experiment/f'after_{a.case}.npz') as z:
        e.data.qpos[:] = z['qpos'][-1]
        e.data.qvel[:] = z['qvel'][-1]
        e.data.time = float(z['time_s'][-1])
    mujoco.mj_fwdPosition(e.model, e.data)
    contacts = []
    for contact in e.data.contact:
        ba, bb = [int(e.model.geom_bodyid[g]) for g in (contact.geom1,contact.geom2)]
        ia, ib = ba in e.robot_bodies, bb in e.robot_bodies
        if ia == ib:
            continue
        body, robot, external = (ba,contact.geom1,contact.geom2) if ia else (bb,contact.geom2,contact.geom1)
        group = e.contact_groups.get(body,'other_robot')
        contacts.append(dict(group=group,robot_geom=e.model.geom(robot).name,external_geom=e.model.geom(external).name,
                             distance_m=float(contact.dist),position_world_m=contact.pos.tolist()))
    foot = e.model.body('left_ankle_roll_link').id
    for geom in range(e.model.ngeom):
        name = e.model.geom(geom).name or ''
        if name == 'rung_01': e.model.geom_rgba[geom] = [.9,.22,.16,1.]
        if int(e.model.geom_bodyid[geom]) == foot: e.model.geom_rgba[geom] = [1.,.65,.12,1.]
    e.model.vis.quality.offsamples = 0
    renderer = mujoco.Renderer(e.model, height=480, width=640)
    options = mujoco.MjvOption()
    options.geomgroup[3], options.geomgroup[4] = 0,1
    canvas = Image.new('RGB',(1280,650),(16,24,32))
    try:
        for i, (lookat,distance,azimuth,elevation) in enumerate([
                ([.26,0.,1.25],2.6,65.,-8.),
                (e.data.xpos[foot].copy(),.65,65.,-8.)]):
            camera = mujoco.MjvCamera()
            camera.lookat[:] = lookat
            camera.distance,camera.azimuth,camera.elevation = distance,azimuth,elevation
            renderer.update_scene(e.data,camera,scene_option=options)
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION]=False
            canvas.paste(Image.fromarray(renderer.render()),(640*i,70))
    finally:
        renderer.close()
        env.close()
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',23)
    small = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18)
    force = metric['last_two_recorded_seconds_group_force']['left_foot']['norm_n']
    draw.text((18,10),f"新杆位 B03：左脚摆动接触失败 | {metric['failure_time_s']:.3f} s",font=font,fill='white')
    draw.text((18,42),'左：保存的终止姿态   右：左脚局部；橙色为左脚，红色为目标横杆',font=small,fill=(170,210,230))
    draw.text((18,565),f"2000 Hz 原动力学记录：左脚接触合力 {force:.2f} N；阶段 {metric['inferred_terminal_controller_phase']}",font=font,fill='white')
    draw.text((18,607),'几何图来自保存状态；重算接触几何不用于冒充原求解器接触力。失败未被剔除。',font=small,fill=(170,210,230))
    canvas.save(a.output/'failure_geometry.png')
    report = dict(case=a.case,contacts_at_saved_terminal_state=contacts,telemetry=metric,
        trajectory_sha256=sha(a.experiment/f'after_{a.case}.npz'), telemetry_audit_sha256=sha(a.telemetry/'failure_telemetry_audit.json'),
        preview_sha256=sha(a.output/'failure_geometry.png'),source_sha256=sha(__file__),
        scope='Saved-state geometry inspection, not a new rollout. Contact geometry is post-integration; measured force is from the original solver telemetry. Colors are visualization only.')
    (a.output/'summary.json').write_text(json.dumps(report,indent=2))
    (a.output/'source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(dict(contacts=[row for row in contacts if row['group']=='left_foot'])))


if __name__ == '__main__':
    main()
