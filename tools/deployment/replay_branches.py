#!/usr/bin/env python3
"""Compare AMP/SMP Python control outputs offline, without DDS or robot I/O.

Each checkout runs in a separate process. The adapter initializes only Passive,
FixedPose, AMP and SMP, then calls that checkout's unmodified runtime.step and
FSM.run. Runtime constructors are deliberately bypassed: older constructors
unconditionally create DDS publishers. This is state-input replay, not physics.
"""
import argparse
import builtins
import hashlib
import importlib.abc
import importlib.util
import inspect
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import types

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
ALLOWED_STATES = ('PASSIVE', 'FIXEDPOSE', 'SKILL_AMP', 'SKILL_SMP_RECOVERY')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_input(path, initial_state):
    """Strict inputs: never invent unrecorded IMU, button, tick or timing data."""
    path = Path(path).resolve(strict=True)
    if path.suffix == '.bin':
        meta = json.loads(path.with_suffix('.json').read_text())
        layout = meta['fields']
        width = sum(layout.values())
        if width != meta['record_dim']:
            raise ValueError('Logger layout width disagrees with metadata')
        raw = np.fromfile(path, dtype=np.float32)
        if len(raw) % width:
            raise ValueError('Partial logger frame')
        raw, values, offset = raw.reshape(-1,width), {}, 0
        for name, size in layout.items():
            values[name], offset = raw[:,offset:offset+size], offset+size
        words = values['state_tick_words']
        if (not np.isfinite(words).all() or np.any(words < 0) or np.any(words > 65535)
                or np.any(words != np.floor(words))):
            raise ValueError('Missing/invalid state tick words')
        words = words.astype(np.uint64)
        values['tick'] = words[:,0]+(words[:,1] << np.uint64(16))
        values['imu_quat_wxyz'] = values['pelvis_quat_w']
        values['imu_gyro'] = values['ang_vel']
        origin = {'kind':'recorded_log', 'metadata':meta,
                  'metadata_sha256':sha256(path.with_suffix('.json'))}
    else:
        with np.load(path,allow_pickle=False) as archive:
            values = {key:archive[key].copy() for key in archive.files if key != 'provenance_json'}
            origin = json.loads(str(archive['provenance_json'].item()))
        if origin.get('kind') not in ('recorded_log','synthetic_diagnostic'):
            raise ValueError('NPZ requires explicit recorded_log or synthetic_diagnostic provenance')
    required = {'q':29,'dq':29,'imu_quat_wxyz':4,'imu_gyro':3,'remote_raw':24,
                'time_s':1,'tick':1,'state_age_ms':1}
    n = len(values['q'])
    if n < 1:
        raise ValueError('Empty input')
    arrays = {}
    for name, width in required.items():
        value = np.asarray(values[name]).reshape(n,width)
        if not np.isfinite(value).all():
            raise ValueError('Missing/nonfinite replay field: '+name)
        arrays[name] = value
    if np.any(np.diff(arrays['time_s'][:,0]) <= 0) or np.any(arrays['state_age_ms'] < 0):
        raise ValueError('Times must increase and reception ages must be nonnegative')
    remote = arrays['remote_raw']
    if np.any(remote != np.floor(remote)) or np.any((remote < 0)|(remote > 255)):
        raise ValueError('remote_raw must contain bytes')
    if np.any(arrays['tick'] < 0) or np.any(arrays['tick'] != np.floor(arrays['tick'])):
        raise ValueError('tick must be a nonnegative integer')
    for packet in remote.astype(np.uint8):
        mask = struct.unpack('<H',packet[2:4].tobytes())[0]
        # B/R1/R2/up/down can enter policies deliberately outside this tool.
        if mask & sum(1<<key for key in (0,4,9,12,14)):
            raise ValueError('Replay requests a policy outside Passive/FixedPose/AMP/SMP')
        if not np.isfinite(struct.unpack('<5f',packet[4:24].tobytes())).all():
            raise ValueError('Nonfinite remote axes')
    if np.any(np.abs(np.linalg.norm(arrays['imu_quat_wxyz'],axis=1)-1) > .05):
        raise ValueError('Invalid pelvis quaternion')
    arrays['remote_raw'] = remote.astype(np.uint8)
    origin.update(input_path=str(path),input_sha256=sha256(path),frames=n,
                  initial_state=initial_state,
                  initialization='explicit policy enter/reset at first input; original hidden history is not reconstructed',
                  scope='same recorded state/button/time inputs; no physics or closed-loop performance claim')
    return arrays, origin


def make_diagnostic(path,scenario='mixed'):
    """Deliberately synthetic inputs exercise priority, warmup and stale input."""
    n = 320 if scenario == 'nominal' else (280 if scenario == 'mixed' else 170)
    q0 = yaml.safe_load((ROOT/'policy/amp/config/amp.yaml').read_text())['default_dof_pos']
    q = np.tile(q0,(n,1)).astype(np.float32)
    q += .01*np.sin(np.arange(n)[:,None]*.1+np.arange(29)[None,:])
    dq = .05*np.cos(np.arange(n)[:,None]*.1+np.arange(29)[None,:])
    remote = np.zeros((n,24),np.uint8)
    buttons = {5:[2],110:[8],140:[11],180:[6],190:[2],195:[2,11],
               210:[6,11],220:[2],230:[8,11],270:[3]}
    if scenario == 'nominal':
        buttons = {5:[2],110:[8],140:[2],250:[11],280:[2]}
    elif scenario == 'f1':
        buttons = {5:[2],110:[8],140:[11],155:[6]}
    elif scenario == 'stale':
        buttons = {5:[2],110:[8]}
    elif scenario in ('start_y','a_y'):
        buttons = {5:[2],110:[2 if scenario=='start_y' else 8,11]}
    for i,keys in buttons.items():
        remote[i,2:4] = np.frombuffer(struct.pack('<H',sum(1<<key for key in keys)),np.uint8)
    for i in range(112,140):
        remote[i,20:24] = np.frombuffer(struct.pack('<f',.4),np.uint8)
    age = np.zeros(n)
    stale_frames = list(range(250,255)) if scenario == 'mixed' else (list(range(125,130)) if scenario=='stale' else [])
    age[stale_frames] = 250.
    ticks = np.arange(1,n+1)
    zero_frames = [258] if scenario=='mixed' else ([134] if scenario=='stale' else [])
    ticks[zero_frames] = 0
    np.savez_compressed(path,q=q,dq=dq,imu_quat_wxyz=np.tile([1.,0,0,0],(n,1)),
        imu_gyro=np.tile([.01,.02,.03],(n,1)),remote_raw=remote,
        time_s=np.arange(n)*.02,tick=ticks,state_age_ms=age,
        provenance_json=json.dumps({'kind':'synthetic_diagnostic',
            'description':'Analytic states and scripted buttons, not a real robot log',
            'scenario':scenario,'events':buttons,'state_stale_frames':stale_frames,
            'tick_zero_frames':zero_frames}))


def contract(root, kind, profile, model):
    rel = 'policy/amp/config/amp.yaml' if kind == 'amp' else 'policy/smp_recovery/config/smp_recovery.yaml'
    cfg = yaml.safe_load((root/rel).read_text())
    if 'model_profiles' in cfg:
        if profile not in cfg['model_profiles']:
            raise ValueError('{} does not have {} profile {}'.format(root,kind,profile))
        selected = cfg['model_profiles'][profile]
    elif kind == 'amp' and profile == 'amp_0309_1':
        selected = {'model_path':cfg['model_path']}
    else:
        raise ValueError('Baseline has no explicitly selected profile '+profile)
    model = Path(model).resolve(strict=True)
    actual = sha256(model)
    expected = selected.get('model_sha256')
    if expected is None:
        # Legacy AMP has no hash in YAML: pin to its tracked model bytes.
        expected = sha256(root/'policy/amp/model'/selected['model_path'])
    if actual != expected:
        raise ValueError('{} {} weight hash differs from profile'.format(kind,profile))
    effective = dict(cfg,**selected)
    effective.pop('model_profiles',None)
    effective.pop('profile',None)
    effective.pop('model_path',None)
    effective.pop('model_sha256',None)
    if kind == 'amp':
        cfg.update(model_path=str(model),model_sha256=actual)
        if 'model_profiles' in cfg:
            cfg['model_profiles'][profile] = dict(selected,model_path=str(model))
    else:
        cfg['model_profiles'][profile] = dict(selected,model_path=str(model))
    return rel, cfg, {'profile':profile,'model_path':str(model),'model_sha256':actual,
                      'config_sha256':sha256(root/rel),'effective':effective}


class NoDDS(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Substitute transport imports; every transport constructor raises."""
    attempts = []

    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(('cyclonedds','unitree_sdk2py')):
            raise ImportError('Transport import forbidden during offline replay: '+fullname)
        if fullname.endswith('_dds'):
            return importlib.util.spec_from_loader(fullname,self)

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        module.__path__ = []
        def forbidden(*args,**kwargs):
            self.attempts.append(module.__name__)
            raise RuntimeError('DDS/publisher construction forbidden: '+module.__name__)
        module.__getattr__ = lambda name: forbidden
        module.apply_diagnostics = lambda state,diagnostics: None


def run_worker(request_path):
    request = json.loads(Path(request_path).read_text())
    root, output = Path(request['root']), Path(request['output'])
    sys.path.insert(0,str(root))
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    sys.dont_write_bytecode = True
    sys.meta_path.insert(0,NoDDS())
    # Only explicit test configuration; ambient profile/env cannot affect replay.
    os.environ['LYD_DEPLOY_CONFIG'] = request['deploy_config']
    os.environ['SMP_RECOVERY_PROFILE'] = request['smp_profile']
    original_open = builtins.open
    redirects = request['redirects']
    def open_config(path,*args,**kwargs):
        if isinstance(path,(str,os.PathLike)):
            path = redirects.get(str(Path(path).resolve()),path)
        return original_open(path,*args,**kwargs)
    builtins.open = open_config
    import torch
    torch.set_num_threads(1)
    import onnxruntime as ort
    original_session = ort.InferenceSession
    def cpu_session(path,options=None,*args,**kwargs):
        options = options or ort.SessionOptions()
        options.intra_op_num_threads,options.inter_op_num_threads = 1,1
        kwargs['providers'] = ['CPUExecutionProvider']
        return original_session(path,options,*args,**kwargs)
    ort.InferenceSession = cpu_session
    from common.ctrlcomp import StateAndCmd,PolicyOutput
    from common.remote_controller import RemoteController
    from common.utils import FSMStateName
    from FSM.FSM import FSM,FSMMode
    from policy.passive.PassiveMode import PassiveMode
    from policy.fixedpose.FixedPose import FixedPose
    from policy.amp.Amp import Amp
    from policy.smp_recovery.SmpRecovery import SmpRecovery
    from bridge.python.policy_runtime import PolicyRuntime
    state, command = StateAndCmd(29),PolicyOutput(29)
    fsm = FSM.__new__(FSM)
    fsm.state_cmd,fsm.policy_output = state,command
    fsm.sim_counter,fsm.FSMmode = 0,FSMMode.NORMAL
    fsm.passive_mode = PassiveMode(state,command)
    fsm.fixed_pose_1 = FixedPose(state,command)
    fsm.amp_policy = Amp(state,command)
    fsm.smp_recovery_policy = SmpRecovery(state,command)
    for name in ('loco_policy','beyondmimic_policy','beyondmimic_mj_policy','standup_mj_policy',
                 'pinocchio_1_6_mj_policy','freekick_policy','php_policy','acp_policy','scalebfm_policy'):
        setattr(fsm,name,None)
    runtime = PolicyRuntime.__new__(PolicyRuntime)
    runtime.state_cmd,runtime.policy_output,runtime.FSM_controller = state,command,fsm
    runtime.num_joints,runtime.control_dt = 29,.02
    runtime.remote_controller = RemoteController()
    runtime.config = types.SimpleNamespace(bridge_read_only=True)
    runtime.connect,runtime.php_enabled,runtime.acp_enabled = False,False,False
    runtime.php_config,runtime.acp_config = {'enabled':False},{'enabled':False}
    runtime._previous_step_time,runtime._was_stale = None,False
    runtime._log_states,runtime._logger,runtime._log_step = set(),None,0
    runtime._cmd_seq,runtime.exit_requested = 0,False
    runtime._acp_trace_enabled = False
    for name in ('_prev_right_pressed','_prev_left_pressed','_prev_l2_right_pressed','_prev_l2_left_pressed'):
        setattr(runtime,name,False)
    runtime.diagnostics_sub = types.SimpleNamespace(matching=lambda tick:None)
    runtime._apply_perception_state = lambda:None
    if hasattr(PolicyRuntime,'_apply_acp_state'):
        from deploy_real.scalebfm_input import RealPolicyInput
        runtime.policy_input = RealPolicyInput(None,php_enabled=False,acp_enabled=False)
    with np.load(request['input'],allow_pickle=False) as archive:
        data = {key:archive[key] for key in archive.files}
    clock = [1000.]
    import time
    time.monotonic,time.time,time.perf_counter = (lambda:clock[0]),(lambda:clock[0]),(lambda:clock[0])
    runtime._log_start = clock[0]
    entries,records = [],[]
    policies = [fsm.passive_mode,fsm.fixed_pose_1,fsm.amp_policy,fsm.smp_recovery_policy]
    for policy in policies:
        enter = policy.enter
        def traced_enter(enter=enter,name=policy.name.name):
            entries.append(name)
            return enter()
        policy.enter = traced_enter
    fsm.cur_policy = {p.name.name:p for p in policies}[request['initial_state']]
    # Same measured initial state and explicit enter/reset in both processes.
    initial = types.SimpleNamespace(q=data['q'][0],dq=data['dq'][0],
        imu_quat_wxyz=data['imu_quat_wxyz'][0],imu_gyro=data['imu_gyro'][0])
    runtime._apply_robot_state(initial)
    fsm.cur_policy.enter()
    entries.clear()
    for i in range(len(data['q'])):
        clock[0] = 1000.+float(data['time_s'][i,0]-data['time_s'][0,0])
        age = float(data['state_age_ms'][i,0])
        tick = int(data['tick'][i,0])
        # BridgeStateSubscriber.latest uses receipt age > stale_ms -> tick=0.
        timeout = age > request['stale_ms']
        bridge = types.SimpleNamespace(q=data['q'][i],dq=data['dq'][i],
            imu_quat_wxyz=data['imu_quat_wxyz'][i],imu_gyro=data['imu_gyro'][i],
            remote_raw=data['remote_raw'][i].tolist(),tick=0 if timeout else tick,
            _local_received_monotonic=clock[0]-age/1000.)
        runtime.bridge_state_sub = types.SimpleNamespace(latest=lambda:bridge)
        before,mode_before = fsm.cur_policy.name.name,fsm.FSMmode.name
        entries.clear()
        frame = runtime.step(bridge) if 'bridge_state' in inspect.signature(runtime.step).parameters else runtime.step()
        q,kp,kd = [np.asarray(value,dtype=np.float32).copy() for value in (frame.q_des,frame.kp,frame.kd)]
        if any(value.shape != (29,) or not np.isfinite(value).all() for value in (q,kp,kd)):
            raise ValueError('Invalid replay output at frame '+str(i))
        records.append(dict(frame=i,time_s=float(data['time_s'][i,0]),state_before=before,
            state=fsm.cur_policy.name.name,mode_before=mode_before,mode=fsm.FSMmode.name,
            executed_state=-1 if bridge.tick==0 else int(getattr(command,'executed_fsm_state',-1)),entered=list(entries),
            request_damping=bool(frame.request_damping),exit_requested=bool(frame.exit_requested),
            damping_output=bool(np.all(kp==0) and np.all(kd>0)),
            state_timeout=bool(timeout or tick==0),command_seq=int(frame.seq),
            qtarget=q.tolist(),kp=kp.tolist(),kd=kd.tolist()))
        if frame.exit_requested:
            break  # Same outer-loop exit condition as deploy_policy.main.
    output.write_text(json.dumps({'records':records,'dds_construction_attempts':NoDDS.attempts,
                                 'runtime_constructor_called':False},indent=2)+'\n')


def compare_records(baseline,candidate,atol):
    if len(baseline) != len(candidate):
        raise ValueError('Replay frame counts differ')
    fields = ('state_before','state','mode_before','mode','executed_state','entered',
              'request_damping','exit_requested','damping_output','state_timeout','command_seq')
    differences,maxima = [],dict(qtarget=0.,kp=0.,kd=0.)
    for left,right in zip(baseline,candidate):
        changed = {key:{'baseline':left[key],'candidate':right[key]} for key in fields if left[key] != right[key]}
        for key in maxima:
            delta = np.abs(np.asarray(left[key])-right[key])
            maxima[key] = max(maxima[key],float(delta.max()))
            if np.any(delta > atol):
                changed[key] = {'max_abs':float(delta.max()),'joint_indices':np.flatnonzero(delta>atol).tolist()}
        if changed:
            differences.append({'frame':left['frame'],'time_s':left['time_s'],'fields':changed})
    return {'equal':not differences,'first_divergence':differences[0] if differences else None,
            'differing_frames':len(differences),'max_abs':maxima,'differences':differences}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',type=Path,help=argparse.SUPPRESS)
    parser.add_argument('--make-diagnostic',type=Path,help='Write explicitly synthetic NPZ inputs and exit')
    parser.add_argument('--scenario',choices=('mixed','nominal','f1','stale','start_y','a_y'),default='mixed')
    parser.add_argument('--baseline-root',type=Path)
    parser.add_argument('--candidate-root',type=Path,default=ROOT)
    parser.add_argument('--input',type=Path)
    parser.add_argument('--initial-state',choices=ALLOWED_STATES)
    parser.add_argument('--amp-profile')
    parser.add_argument('--smp-profile')
    parser.add_argument('--amp-model',type=Path)
    parser.add_argument('--smp-model',type=Path)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--stale-ms',type=float,default=200.)
    parser.add_argument('--atol',type=float,default=1e-6)
    args = parser.parse_args()
    if args.worker:
        return run_worker(args.worker)
    if args.make_diagnostic:
        return make_diagnostic(args.make_diagnostic,args.scenario)
    for name in ('baseline_root','input','initial_state','amp_profile','smp_profile','amp_model','smp_model','output_dir'):
        if getattr(args,name) is None:
            parser.error('--'+name.replace('_','-')+' is required')
    if not np.isfinite([args.stale_ms,args.atol]).all() or args.stale_ms <= 0 or args.atol < 0:
        parser.error('Invalid stale-ms/atol')
    arrays,provenance = load_input(args.input,args.initial_state)
    output = args.output_dir.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error('--output-dir must be a new or empty directory; existing evidence is preserved')
    output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output/'input.npz',**arrays)
    result = {'input':provenance,'atol':args.atol,'stale_ms':args.stale_ms,
              'adapter_source_sha256':sha256(__file__),
              'status':'ready_for_real_log' if provenance['kind']=='synthetic_diagnostic' else 'recorded_segment_compared',
              'hardware_baseline_verified':False,
              'scope':'bridge Python runtime.step + FSM.run + AMP/SMP only; direct Python hardware entry and historical robot chain are not replayed',
              'adapter':{'skipped_initialization':['PolicyRuntime.__init__','FSM.__init__',
                  'DDS subscribers/publishers','perception state/publishing','logging','unselected policy loading'],
                  'actual_methods':['PolicyRuntime.step','PolicyRuntime._set_remote_from_bridge',
                    'PolicyRuntime._apply_robot_state','PolicyRuntime._apply_remote_commands',
                    'FSM.run','FSM.get_next_policy','PassiveMode enter/run/exit/checkChange',
                    'FixedPose enter/run/exit/checkChange','Amp enter/run/exit/checkChange',
                    'SmpRecovery enter/run/exit/checkChange','RealPolicyInput.update (candidate only)'],
                  'initialized_fields':'state/output; 4 real policies; FSM normal mode; input edge latches; remote decoder; sequence; clock/telemetry; disabled ACP/PHP flags; no logger; replay state provider',
                  'state_timeout_adapter':'BridgeStateSubscriber receipt-age > stale_ms converted to tick=0; DDS receipt/poll loop itself is not executed',
                  'termination':'stop after first exit_requested, as the deploy_policy outer loop; publisher/finalizer is never executed'},
              'transport':'all DDS/SDK imports blocked; no runtime constructors or publishers',
              'disabled_policies':['ACP','PHP','ScaleBFM','Loco','BeyondMimic','FreeKick'],
              'cpp_command_timeout':'not executed or claimed: Python state replay does not run the C++ bridge',
              'branches':{}}
    runs = []
    for label,root in (('baseline',args.baseline_root),('candidate',args.candidate_root)):
        root = root.resolve(strict=True)
        redirects,contracts = {},{}
        for kind,profile,model in (('amp',args.amp_profile,args.amp_model),('smp',args.smp_profile,args.smp_model)):
            rel,cfg,contracts[kind] = contract(root,kind,profile,model)
            config_path = output/(label+'_'+kind+'.yaml')
            config_path.write_text(yaml.safe_dump(cfg))
            redirects[str(root/rel)] = str(config_path)
        deploy = output/(label+'_deploy.yaml')
        deploy.write_text(yaml.safe_dump(dict(amp={'profile':args.amp_profile},smp={'profile':args.smp_profile},
            scalebfm={'enabled':False},php={'enabled':False,'real':{'enabled':False}},
            acp={'enabled':False,'real':{'enabled':False}})))
        request = dict(root=str(root),output=str(output/(label+'.json')),input=str(output/'input.npz'),
                       initial_state=args.initial_state,stale_ms=args.stale_ms,smp_profile=args.smp_profile,
                       redirects=redirects,deploy_config=str(deploy))
        request_path = output/(label+'_request.json')
        request_path.write_text(json.dumps(request))
        with (output/(label+'.log')).open('w') as log:
            subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker',str(request_path)],
                           stdout=log,stderr=subprocess.STDOUT,check=True,cwd=root)
        run = json.loads((output/(label+'.json')).read_text())
        if run['dds_construction_attempts']:
            raise RuntimeError('Transport construction attempted')
        runs.append(run['records'])
        deployment = root/'config/deploy.yaml'
        defaults = yaml.safe_load(deployment.read_text()) if deployment.exists() else {}
        smp_stock = yaml.safe_load((root/'policy/smp_recovery/config/smp_recovery.yaml').read_text())
        result['branches'][label] = dict(root=str(root),
            commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
            tracked_dirty=bool(subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=root,text=True)),
            contracts=contracts,
            configured_defaults={'amp':defaults.get('amp',{}).get('profile','amp_0309_1'),
                                 'smp':defaults.get('smp',{}).get('profile',smp_stock.get('profile')),
                                 'note':'checkout configuration only; historical robot environment overrides are unverified'},
            source_sha256={rel:sha256(root/rel) for rel in ('bridge/python/policy_runtime.py','FSM/FSM.py',
                'policy/amp/Amp.py','policy/smp_recovery/SmpRecovery.py',
                'policy/fixedpose/FixedPose.py','policy/passive/PassiveMode.py',
                'common/ctrlcomp.py','common/rotation_helper.py','common/remote_controller.py')})
    result['comparison'] = compare_records(*runs,args.atol)
    (output/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'equal':result['comparison']['equal'],'first_divergence':result['comparison']['first_divergence'],
                      'input_kind':provenance['kind'],'report':str(output/'comparison.json')}))


if __name__ == '__main__':
    main()
