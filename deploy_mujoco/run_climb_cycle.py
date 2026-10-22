"""Execute a declared full-cycle reference through motor torques in MuJoCo."""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import time

import mujoco
import numpy as np
from omegaconf import OmegaConf

from common.path_config import PROJECT_ROOT
from training.wall_support_env import WallSupportEnv
from training.climb_cycle import CycleTorqueController
from training.hand_transfer import smooth
from training.climb_robustness import apply_rung_offsets, ReleaseInterlock
from training.release_feasibility import allocate_before_release
from training.unload_protection import current_pose_hold
from training.unload_lookahead import UnloadLookahead
from training.tracking_relaxation import TrackingRelaxationController
from training.contact_frame_controller import ContactFrameController
from training.probing_contact_frame_controller import ProbingContactFrameController
from training.contact_landing import ContactLanding
from training.cycle_clock import cycle_index

ROOT=Path(PROJECT_ROOT)


def assess_cycle_end(rows,contacts,end,index):
    tail=[r for r in rows if end-2.<r['time_s']<=end+1e-7]
    expected={g:f'rung_{(4 if "hook" in g else 0)+index:02d}' for g in
              ['left_hook','right_hook','left_foot','right_foot']}
    loaded=[c for item in contacts if end-2.<item['time_s']<=end+1e-7 for c in item['contacts']
            if np.linalg.norm(c['force_on_robot_world_n'])>1.]
    checks=dict(complete_window=len(tail)>=99,
                final_height=bool(tail) and all(abs(r['base_rise_m']-.25*index)<.01 for r in tail),
                final_four_supports=bool(tail) and all(min(r[g+'_vertical_n'] for g in expected)>10. for r in tail),
                correct_final_rungs=bool(loaded) and all(c['group'] in expected and c['external_geom']==expected[c['group']] for c in loaded),
                closed_joint_posture=bool(tail) and all(r['periodic_joint_error_rad']<.04 for r in tail))
    return dict(cycle=index,end_s=float(end),passed=all(checks.values()),checks=checks,
                height_target_m=.25*index,expected_rungs=expected,sample_count=len(tail))


def run_climb_cycle(cfg):
    out=ROOT/str(cfg.output_dir)
    if out.exists():raise FileExistsError(out)
    plan_path=ROOT/str(cfg.plan)
    plan=np.load(plan_path)
    plan_audit_path=plan_path.parent/'audit.json'
    plan_audit=json.loads(plan_audit_path.read_text()) if plan_audit_path.exists() else None
    if not cfg.diagnostic and (not plan_audit or not plan_audit['passed']):
        raise ValueError('A passed dense reference audit is required for a non-diagnostic run')
    plan_hash=hashlib.sha256(plan_path.read_bytes()).hexdigest()
    plan_manifest_path=plan_path.parent/'manifest.json'
    plan_manifest=json.loads(plan_manifest_path.read_text()) if plan_manifest_path.exists() else {}
    if not cfg.diagnostic and plan_manifest.get('plan_sha256')!=plan_hash:
        raise ValueError('Reference hash differs from the audited plan')
    horizon=float(plan['time'][-1]) if cfg.seconds is None else min(float(cfg.seconds),float(plan['time'][-1]))
    cycle_ends=np.asarray(plan.get('cycle_ends_s',[plan['time'][-1]]),dtype=float)
    if cycle_ends.ndim!=1 or not len(cycle_ends) or np.any(np.diff(np.r_[0.,cycle_ends])<=0) or abs(cycle_ends[-1]-plan['time'][-1])>1e-8:
        raise ValueError('Cycle endpoints must partition the entire reference duration')
    used=plan['time']<=horizon
    if max(plan['errors'][used])>.002 and not cfg.diagnostic:
        raise ValueError('Reference IK residual exceeds 2 mm; refusing execution')
    out.mkdir(parents=True)
    sources={}
    extra_sources=['training/tracking_relaxation.py','training/contact_frame_controller.py','training/probing_contact_frame_controller.py','training/contact_landing.py','training/cycle_clock.py','tools/audit_contact_landing.py']
    for path in extra_sources:
        dest=out/'source_snapshot'/path;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/path,dest)
        sources[path]=hashlib.sha256(dest.read_bytes()).hexdigest()
    for path in ['training/climb_cycle.py','training/climb_robustness.py','training/release_feasibility.py','training/unload_protection.py','training/unload_lookahead.py','training/climb_env.py','training/wall_support_env.py','deploy_mujoco/run_climb_cycle.py','deploy_mujoco/config/mujoco_climb_cycle.yaml','deploy_mujoco/config/mujoco_climb_cycle_optimized.yaml']:
        dest=out/'source_snapshot'/path;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/path,dest)
        sources[path]=hashlib.sha256(dest.read_bytes()).hexdigest()
    shutil.copy2(plan_path,out/'reference.npz')
    if plan_audit_path.exists():shutil.copy2(plan_audit_path,out/'reference_audit.json')
    (out/'manifest.json').write_text(json.dumps(dict(bundle=str(ROOT/'outputs/wall_support_bundle_pd15_v2'),
        sources=sources,plan_sha256=plan_hash,config=OmegaConf.to_container(cfg),
        mujoco_version=mujoco.__version__,numpy_version=np.__version__),indent=2))
    env=WallSupportEnv(ROOT/'outputs/wall_support_bundle_pd15_v2')
    env.reset();m,d=env.model,env.data
    offsets=OmegaConf.to_container(cfg.get('rung_offsets',OmegaConf.create({})))
    variant=apply_rung_offsets(env,offsets)
    (out/'scene_variant.json').write_text(json.dumps(variant,indent=2))
    guard=ReleaseInterlock(env) if cfg.get('release_guard',False) else None
    if cfg.get('allocation_failure_hold',False) and guard is None:
        raise ValueError('Current-pose failure hold requires the contact confirmation guard')
    lookahead_seconds=float(cfg.get('unload_lookahead_seconds',0.))
    frame_from_transfer=bool(cfg.get('contact_frame_from_transfer',False))
    reset_transfer_each_cycle=bool(cfg.get('reset_contact_frame_each_cycle',False))
    if reset_transfer_each_cycle and not frame_from_transfer:
        raise ValueError('Per-cycle contact establishment requires continued contact-frame mode')
    landing_enabled=bool(cfg.get('contact_landing',False))
    if landing_enabled and (not frame_from_transfer or cfg.resume_from or len(cycle_ends)!=1 or guard):
        raise ValueError('Landing prototype requires one uninterrupted cycle, continued contact frames, and no separate release guard')
    if frame_from_transfer and cfg.resume_from:
        raise ValueError('Continued contact-frame resume requires controller state and is not supported')
    use_transfer_frame=bool(cfg.get('contact_frame_transfer',False) or frame_from_transfer)
    if frame_from_transfer and cfg.get('contact_frame_transfer',False):
        raise ValueError('Select transfer-only or continued contact-frame mode')
    if use_transfer_frame and cfg.get('contact_frame_controller',False):
        raise ValueError('Select full contact-frame or transfer-only mode')
    if (cfg.get('contact_frame_controller',False) or use_transfer_frame) and cfg.get('tracking_relaxation',False):
        raise ValueError('Contact-frame and tracking-relaxation combination has not been validated')
    if lookahead_seconds and (cfg.get('tracking_relaxation',False) or cfg.get('contact_frame_controller',False) or use_transfer_frame):
        raise ValueError('Relaxed tracking and original-controller lookahead cannot be combined')
    if lookahead_seconds and not cfg.get('allocation_failure_hold',False):
        raise ValueError('Lookahead requires allocation_failure_hold')
    lookahead=UnloadLookahead(env,offsets,lookahead_seconds) if lookahead_seconds else None
    next_prediction=0.;prediction_log=[]
    blocked_since=None;hold_q=None;hold_active=None;hold_phase=None
    last_q=None;last_active=None;last_command_phase=None
    env.initial=plan['qpos'][0].copy();d.qpos[:]=env.initial
    mujoco.mj_forward(m,d)
    start_time=0.
    if cfg.resume_from:
        saved=np.load(ROOT/str(cfg.resume_from))
        d.qpos[:]=saved['qpos'][-1];d.qvel[:]=saved['qvel'][-1]
        start_time=float(saved['time_s'][-1]);d.time=start_time;mujoco.mj_forward(m,d)
    env.initial_hands=d.site_xpos[env.hand_sites].copy()
    gain_mode=str(cfg.get('leg_gain_mode','constant'))
    if gain_mode not in ('constant','settle'):raise ValueError('Unknown leg_gain_mode')
    normal_gain=float(cfg.get('joint_kp',100.));settle_gain=normal_gain if cfg.get('leg_kp') is None else float(cfg.leg_kp)
    controller_class=TrackingRelaxationController if cfg.get('tracking_relaxation',False) else CycleTorqueController
    if cfg.get('contact_frame_controller',False):controller_class=ContactFrameController
    controller=controller_class(env,joint_kp=normal_gain,torque_weight=float(cfg.get('torque_weight',0.)),leg_kp=normal_gain if gain_mode=='settle' else cfg.get('leg_kp'))
    baseline_controller=controller
    transfer_class=ProbingContactFrameController if landing_enabled else ContactFrameController
    transfer_controller=transfer_class(env,joint_kp=normal_gain,torque_weight=float(cfg.get('torque_weight',0.)),leg_kp=normal_gain) if use_transfer_frame else None
    landing=ContactLanding(env) if landing_enabled else None
    online_qrefs=[];online_vrefs=[];online_active=[];online_times=[]
    transfer_entered=False
    controller_cycle=0
    previous_controller=None;controller_switch_events=[]
    phase_gain=controller.leg_kp;phase_start=start_time
    rows=[];states=[d.qpos.copy()];velocities=[d.qvel.copy()];contacts=[];commands=[]
    times=plan['time'];qvalues=plan['qpos'];warnings=d.warning.number.copy()
    reason='complete';other_peak=0.;max_torque_ratio=0.;equilibrium_peak=0.;flight_peak=0.;base_error_peak=0.;self_peak=0.
    phase_events=[];last_phase=None
    wall_start=time.perf_counter();torque=np.zeros(29)
    def reference(t):
        j=min(max(int(np.searchsorted(times,t,side='right')-1),0),len(times)-2)
        dt=times[j+1]-times[j];u=float(np.clip((t-times[j])/dt,0,1))
        q=(1-u)*qvalues[j]+u*qvalues[j+1]
        v=np.zeros(m.nv);v[:3]=(qvalues[j+1,:3]-qvalues[j,:3])/dt
        v[env.vadr]=(qvalues[j+1,env.qadr]-qvalues[j,env.qadr])/dt
        return q,v,plan['active'][j+1],str(plan['phase'][j+1])
    for step in range(round((horizon-start_time)/.005)):
        qref,vref,active,phase=reference(float(d.time))
        if guard:
            guard.sample(float(d.time))
            if blocked_since is None and not guard.allow(float(d.time),active):
                blocked_since=float(d.time)
                hold_q=last_q.copy()
                # Do not allocate load to the unconfirmed new contact while holding.
                confirmed=np.all(np.asarray([v for _,v in guard.history],dtype=bool),axis=0)
                hold_active=last_active & confirmed
                hold_phase=last_phase
                guard.events[-1]['hold_active']=hold_active.tolist()
            if blocked_since is not None:
                if d.time-blocked_since>=.5-1e-8:
                    reason='release_interlock_stop';break
                qref=hold_q.copy();vref=np.zeros(m.nv);active=hold_active;phase=hold_phase
        if transfer_controller:
            current_cycle=cycle_index(float(d.time),cycle_ends)
            if reset_transfer_each_cycle and current_cycle!=controller_cycle:
                transfer_entered=False
                controller_cycle=current_cycle
            transfer_phase=phase in ('weight_transfer','lf_unload','lf_swing','lf_land')
            transfer_entered=transfer_entered or transfer_phase
            controller=transfer_controller if transfer_phase or (frame_from_transfer and transfer_entered) else baseline_controller
        if phase!=last_phase:
            phase_events.append(dict(time_s=float(d.time),phase=phase));last_phase=phase
            phase_start=float(d.time);phase_gain=controller.leg_kp
        if gain_mode=='settle':
            target=settle_gain if phase in ('final_center','final_hold','posture_closure','closed_hold') else normal_gain
            w=smooth(float(np.clip((d.time-phase_start)/3.,0.,1.)))
            controller.leg_kp=(1-w)*phase_gain+w*target
        if controller is not previous_controller:
            controller_switch_events.append(dict(time_s=float(d.time),cycle=cycle_index(float(d.time),cycle_ends)+1,
                phase=phase,mode='contact_frame' if isinstance(controller,(ContactFrameController,ProbingContactFrameController)) else 'fixed_contact_points',
                leg_acceleration_kp=controller.leg_kp))
            previous_controller=controller
        if landing:
            try:qref,vref,active=landing.update(qref,vref,active,phase,controller)
            except RuntimeError as exc:reason=str(exc);break
        if lookahead and blocked_since is None and phase=='lf_unload' and d.time>=next_prediction-1e-8:
            next_prediction=float(d.time)+.05
            prediction=lookahead.predict(env,reference,controller.leg_kp)
            prediction_log.append(dict(time_s=float(d.time),**prediction))
            if prediction['failed']:
                covered=bool(guard.history) and d.time-guard.history[0][0]>=guard.duration-1e-7
                confirmed=np.all(np.asarray([v for _,v in guard.history],dtype=bool),axis=0) if covered else np.zeros(4,dtype=bool)
                before_active=np.asarray(active,dtype=bool).copy()
                try:
                    _,_,hold_q,hold_active=current_pose_hold(controller,active,confirmed)
                except RuntimeError as rejected:
                    prediction_log[-1]['hold_rejection']=str(rejected)
                else:
                    blocked_since=float(d.time);hold_phase=phase
                    guard.events.append(dict(time_s=blocked_since,allowed=False,released=[],
                        event='predicted_tracking_failure',prediction=prediction,
                        active_before=before_active.tolist(),confirmed=confirmed.tolist(),
                        hold_active=hold_active.tolist(),window_s=guard.duration))
                    qref=hold_q.copy();vref=np.zeros(m.nv);active=hold_active
                    phase_events.append(dict(time_s=blocked_since,phase=phase,event='predictive_current_pose_hold'))
        try:
            releasing=last_active is not None and np.any(last_active & ~np.asarray(active,dtype=bool))
            if guard and blocked_since is None and releasing:
                confirmed=np.all(np.asarray([v for _,v in guard.history],dtype=bool),axis=0)
                retained=last_active & confirmed
                torque,info,rejected=allocate_before_release(controller,qref,vref,active,last_q,retained)
                if rejected is not None:
                    blocked_since=float(d.time);hold_q=last_q.copy();hold_active=retained;hold_phase=last_command_phase
                    guard.events[-1].update(allowed=False,hold_active=retained.tolist(),allocation_rejection=rejected)
                    qref=hold_q.copy();vref=np.zeros(m.nv);active=hold_active;phase=hold_phase
                    phase_events.append(dict(time_s=float(d.time),phase=phase,event='release_allocation_rejected'))
                    last_phase=phase;phase_start=float(d.time);phase_gain=controller.leg_kp
            else:torque,info=controller.command(qref,vref,active)
        except RuntimeError as exc:
            np.savez_compressed(out/'failure_state.npz',qpos=d.qpos.copy(),qvel=d.qvel.copy(),
                                qref=qref,vref=vref,active=active,time_s=float(d.time),phase=phase)
            if not cfg.get('allocation_failure_hold',False) or blocked_since is not None:
                reason=str(exc);break
            covered=bool(guard.history) and d.time-guard.history[0][0]>=guard.duration-1e-7
            confirmed=np.all(np.asarray([v for _,v in guard.history],dtype=bool),axis=0) if covered else np.zeros(4,dtype=bool)
            before_active=np.asarray(active,dtype=bool).copy()
            try:
                torque,info,hold_q,hold_active=current_pose_hold(controller,active,confirmed)
            except RuntimeError as hold_failure:
                reason=f'Tracking allocation failed ({exc}); current-pose hold rejected ({hold_failure})';break
            blocked_since=float(d.time);hold_phase=phase
            guard.events.append(dict(time_s=blocked_since,allowed=False,released=[],
                event='tracking_allocation_failure',allocation_rejection=str(exc),
                active_before=before_active.tolist(),confirmed=confirmed.tolist(),
                hold_active=hold_active.tolist(),window_s=guard.duration))
            qref=hold_q.copy();vref=np.zeros(m.nv);active=hold_active
            phase_events.append(dict(time_s=blocked_since,phase=phase,event='current_pose_hold'))
        max_torque_ratio=max(max_torque_ratio,info['torque_ratio'])
        equilibrium_peak=max(equilibrium_peak,info['base_equilibrium_residual'])
        last_q=qref.copy();last_active=np.asarray(active).copy();last_command_phase=phase
        if landing:
            online_qrefs.append(qref.copy());online_vrefs.append(vref.copy());online_active.append(np.asarray(active).copy());online_times.append(float(d.time))
        d.ctrl[env.motors]=torque
        for _ in range(10):
            assert not np.any(d.qfrc_applied) and not np.any(d.xfrc_applied)
            mujoco.mj_step(m,d);mujoco.mj_kinematics(m,d)
            if np.any(d.warning.number>warnings) or not np.isfinite(d.qpos).all():reason='physics_failure'
            error=float(np.linalg.norm(d.qpos[:3]-qref[:3]))
            base_error_peak=max(base_error_peak,error)
            if error>.15:reason='base_tracking_loss'
            for ci,c in enumerate(d.contact):
                ba,bb=int(m.geom_bodyid[c.geom1]),int(m.geom_bodyid[c.geom2])
                ina,inb=ba in env.robot_bodies,bb in env.robot_bodies
                if ina and inb:
                    force=np.zeros(6);mujoco.mj_contactForce(m,d,ci,force)
                    self_peak=max(self_peak,float(np.linalg.norm(force[:3])))
                if ina==inb:continue
                body=ba if ina else bb
                if body not in env.contact_groups:
                    force=np.zeros(6);mujoco.mj_contactForce(m,d,ci,force)
                    other_peak=max(other_peak,float(np.linalg.norm(force[:3])))
                prefix=phase[:2]
                clear_phase=phase.endswith(('_swing','_straighten')) or (prefix in ['lh','rh'] and phase.split('_',1)[1] in ['retreat','raise','approach'])
                if clear_phase and body==controller.ik.bodies.get(prefix):
                    force=np.zeros(6);mujoco.mj_contactForce(m,d,ci,force)
                    flight_peak=max(flight_peak,float(np.linalg.norm(force[:3])))
            if reason!='complete':break
        if step%4==3 or reason!='complete':
            force=env.contact_metrics();contact=force.pop('contacts');force.pop('force_sample')
            row=dict(time_s=float(d.time),phase=phase,controller_mode='contact_frame' if isinstance(controller,(ContactFrameController,ProbingContactFrameController)) else 'fixed_contact_points',leg_acceleration_kp=controller.leg_kp,tracking_scale=info.get('tracking_scale',1.),base_error_m=error,base_rise_m=float(d.qpos[2]-env.initial[2]),
                     other_contact_peak_n=other_peak,flight_contact_peak_n=flight_peak,self_contact_peak_n=self_peak,
                     base_error_peak_m=base_error_peak,
                     periodic_joint_error_rad=float(max(abs(d.qpos[env.qadr]-env.initial[env.qadr]))),
                     max_joint_error_rad=float(max(abs(d.qpos[env.qadr]-qref[env.qadr]))),**force)
            rows.append(row);states.append(d.qpos.copy());velocities.append(d.qvel.copy());commands.append(torque.copy())
            contacts.append(dict(time_s=float(d.time),phase=phase,contacts=contact))
        if step%400==399:
            message=json.dumps(rows[-1]);print(message,flush=True)
            with (out/'runtime.log').open('a',encoding='utf-8') as f:f.write(message+'\n')
        if reason!='complete':break
    if rows:
        with (out/'control_log.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    np.savez_compressed(out/'trajectory.npz',qpos=states,qvel=velocities,torques=commands,time_s=[start_time]+[r['time_s'] for r in rows])
    if landing:
        np.savez_compressed(out/'online_reference.npz',qpos=online_qrefs,qvel=online_vrefs,active=online_active,time_s=online_times)
        (out/'landing_log.json').write_text(json.dumps(landing.records))
    (out/'contact_log.json').write_text(json.dumps(contacts,indent=2))
    (out/'phase_events.json').write_text(json.dumps(phase_events,indent=2))
    (out/'controller_switch_events.json').write_text(json.dumps(controller_switch_events,indent=2))
    (out/'release_events.json').write_text(json.dumps(guard.events if guard else [],indent=2))
    (out/'lookahead_log.json').write_text(json.dumps(prediction_log,indent=2))
    cycle_results=[assess_cycle_end(rows,contacts,float(end),i+1) for i,end in enumerate(cycle_ends)]
    final_checks=cycle_results[-1]['checks']
    checks=dict(full_duration=bool(reason=='complete' and start_time==0. and abs(d.time-times[-1])<1e-7),
                final_height=final_checks['final_height'],
                final_four_supports=final_checks['final_four_supports'],
                correct_final_rungs=final_checks['correct_final_rungs'],
                no_unintended_support=other_peak<1.,clear_swing=flight_peak<1.,base_tracking=base_error_peak<.03,
                no_self_collision=self_peak<1.,
                closed_joint_posture=final_checks['closed_joint_posture'],
                every_cycle_endpoint=all(r['passed'] for r in cycle_results),
                torque_bounds=max_torque_ratio<=1.+1e-5,reference_audit=bool(plan_audit and plan_audit['passed']))
    summary=dict(completed=reason=='complete',full_cycle=bool(reason=='complete' and start_time==0. and abs(d.time-times[-1])<1e-7),
                 passed=bool(all(checks.values())),checks={k:bool(v) for k,v in checks.items()},
                 reason=reason,simulation_time_s=float(d.time),wall_seconds=time.perf_counter()-wall_start,
                 other_contact_peak_n=other_peak,max_torque_ratio=max_torque_ratio,base_equilibrium_residual=equilibrium_peak,
                 flight_contact_peak_n=flight_peak,base_error_peak_m=base_error_peak,
                 self_contact_peak_n=self_peak,
                 release_blocked_at_s=blocked_since,hold_duration_s=0. if blocked_since is None else float(d.time-blocked_since),
                 scene_variant=variant,solver_retries=sum(c.solver_retries for c in [baseline_controller,transfer_controller] if c is not None),
                 solver_fallbacks=sum(c.solver_fallbacks for c in [baseline_controller,transfer_controller] if c is not None),
                 relaxation_steps=getattr(controller,'relaxation_steps',0),minimum_tracking_scale=getattr(controller,'minimum_tracking_scale',1.),
                 landing=None if landing is None else dict(mode=landing.mode,depth_m=landing.depth,confirmed_at_s=landing.confirmed_at,maximum_ik_error=landing.maximum_ik_error,
                    scope='Nominal reference audit does not cover online corrections; separate online-reference audit required'),
                 cycle_results=cycle_results,
                 final=rows[-1] if rows else None,config=OmegaConf.to_container(cfg),trained_policy=False,
                 plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest())
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)
    if lookahead:lookahead.close()
    env.close()
