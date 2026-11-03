"""Exploratory cloud-guidance/local-correction experiment in one synthetic room.

Uses archived lab decision functions with explicit model, sensor, parser,
scheduling and velocity-execution adapters. The local correction baseline is a
prompted pretrained VLM direction replacement, not AsyncVLA/TIC-VLA replication.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import threading
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from PIL import Image
from .cloud_chat import CloudChat
from .lab_native import EpisodeStopped,SOURCE,geometry,navigation
from .refiner_contract import POINT_PROMPT
from .continuous_clock import continuous_s,continuous_ns
from .local_vlm import LocalVLM
from .locomotion import ROOT,Go2Locomotion,build_scene,healthy,POLICY_PATH

OUT=ROOT/'runs/model-experiments'

def pixels(rgb,fovy):
    p=rgb.astype(np.int16)
    ys,xs=np.where((p[:,:,2]>90)&(p[:,:,2]>2*p[:,:,0])&(p[:,:,2]>2*p[:,:,1]))
    if len(xs)<10:return {'visible':False,'bearing':None,'pixels':len(xs)}
    n=(rgb.shape[1]/2-float(xs.mean()))/(rgb.shape[1]/2)
    bearing=math.atan(n*(rgb.shape[1]/rgb.shape[0])*math.tan(math.radians(fovy/2)))
    return {'visible':True,'bearing':bearing,'pixels':len(xs)}

def command_side(cmd):
    return 'LEFT' if cmd[2]>.2 else 'RIGHT' if cmd[2]<-.2 else 'CENTER' if cmd[0]>0 else 'ABSENT'

def direction_command(side):
    return {'left':[0.,0.,.6],'right':[0.,0.,-.6],'center':[.35,0.,0.]}.get(side,[0.,0.,0.])

def useful(cmd,p):
    if not p['visible']:return bool(abs(cmd[2])>.2),'target not visible'
    b=p['bearing']
    if cmd[0]>0:return abs(b)<=math.radians(20),'forward bearing envelope'
    if cmd[2]>.2:return b>math.radians(5),'left-turn bearing envelope'
    if cmd[2]<-.2:return b<-math.radians(5),'right-turn bearing envelope'
    return False,'visible target needs progress'

def height_view(model,data):
    base=model.body('base').id;rotation=data.xmat[base].reshape(3,3)
    yaw=math.atan2(rotation[1,0],rotation[0,0]);forward=np.array([math.cos(yaw),math.sin(yaw)])
    right=np.array([math.sin(yaw),-math.cos(yaw)]);origin=data.xpos[base,:2]
    result=np.empty((16,32),dtype=np.int16);mask=np.array([0,0,0,0,0,1],dtype=np.uint8)
    hit=np.array([-1],dtype=np.int32)
    for d in range(16):
        for c in range(32):
            xy=origin+forward*((d+1)*.24)+right*((c-16)*.24)
            distance=mujoco.mj_ray(model,data,np.array([xy[0],xy[1],3.]),np.array([0.,0.,-1.]),mask,True,-1,hit)
            result[d,c]=-128 if distance<0 else int(np.clip(round((3-distance)*100),-127,127))
    return result,yaw

class Sensors:
    def __init__(self,period):
        self.obs=None;self.condition=threading.Condition();self.closed=False
        self.period=period;self.next_plan=0.;self.planner_snapshot=None
    @property
    def hm_grid(self):return None if self.obs is None else self.obs['grid']
    @property
    def hm_yaw(self):return 0. if self.obs is None else self.obs['yaw']
    def latest_frame_jpeg(self):
        obs=self.obs
        if obs is None:return None
        self.planner_snapshot={k:obs[k] for k in ['frame','acquired_ns','sim_s']}
        return obs['jpeg']
    def wait_map(self,_rx,baseline):
        with self.condition:
            while not self.closed and (self.obs is None or self.obs['frame']==baseline or time.monotonic()<self.next_plan):self.condition.wait(.05)
            if self.closed:raise EpisodeStopped()
            self.next_plan=time.monotonic()+self.period
            return self.obs['frame']
    def update(self,obs):
        with self.condition:self.obs=obs;self.condition.notify_all()
    def close(self):
        with self.condition:self.closed=True;self.condition.notify_all()

def run(args):
    assert shutil.disk_usage(ROOT).free>(8 if args.mode in ['frequent','selective'] else 4)*1024**3, 'Insufficient free-space reserve for this run'
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out=OUT/'rollouts'/f'{stamp}_{args.phase}_{args.mode}_{args.scenario}';out.mkdir(parents=True)
    scene=build_scene(out);tree=ET.parse(scene);world=tree.getroot().find('worldbody')
    red=ET.SubElement(world,'body',name='visual_distractor',mocap='true',pos='3.5 -2 1')
    ET.SubElement(red,'geom',name='visual_distractor',type='box',size='.3 .3 .3',rgba='.9 .03 .03 1',contype='0',conaffinity='0',group='4')
    tree.write(scene,encoding='unicode')
    model=mujoco.MjModel.from_xml_path(str(scene));data=mujoco.MjData(model)
    environment=['floor','far_wall','left_wall','right_wall','blue_target']
    for name in environment:model.geom_group[model.geom(name).id]=5
    options=mujoco.MjvOption();options.geomgroup[4]=1;options.geomgroup[5]=1
    mujoco.mj_resetDataKeyframe(model,data,0);mujoco.mj_forward(model,data)
    gait=Go2Locomotion(model)
    for _ in range(250):gait.update(model,data,[0,0,0]);mujoco.mj_step(model,data)
    target=int(model.body_mocapid[model.body('target').id]);data.mocap_pos[target]=[2.,.7,.22]
    distractor=int(model.body_mocapid[model.body('visual_distractor').id]);mujoco.mj_forward(model,data)
    fovy=float(model.cam_fovy[model.camera('front').id]);s=geometry();sensor=Sensors(args.plan_period)
    metadata={'scope':'exploratory cloud-guidance / prompted local direction replacement prototype',
        'worker_sha256':hashlib.sha256((ROOT/'scripts/vlm_worker.py').read_bytes()).hexdigest(),'parameters':vars(args),'cloud_model':'gpt-4.1-2025-04-14','local_model':args.refiner_model,
        'local_refiner_kind':'pretrained VLM visual direction replacement via normalized-point output; not command-conditioned, trained, or a published refiner reproduction',
        'local_output_format':'point','local_prompt':POINT_PROMPT,'refiner_contract_sha256':hashlib.sha256((ROOT/'experiment/src/robotics_lab/refiner_contract.py').read_bytes()).hexdigest(),'goal':'walk to the blue box',
        'lab_model_reproduction':False,'parser_adapter':'provisional JSON validator pending recovered helper comparison',
        'sensor_adapter':'idealized top-down raycast 16x32 robot-aligned centimetre height view; no LiDAR occlusion/noise model',
        'execution_adapter':'shared continuous desired-velocity interface with 0.35m/s and 0.6rad/s caps and fresh geometry',
        'slow_query_period_s':args.plan_period,'same_validity_gate_for_all_modes':True,
        'frequent_policy':'issue local correction whenever idle','selective_policy':'issue local correction only without a useful active command',
        'camera_hz':10,'physics_hz':250,'gait_hz':50,'fixed_known_color_fixture':True,
        'online_ground_truth_use':'raycast sensor geometry and robot pose only; target identity/position supplied solely by camera pixels',
        'oracle_usage':'scripted target motion and independent scoring',
        'completion':'collision-free upright robot within 0.9m, speed<=0.08m/s for 0.5 simulated seconds, before shared wall-clock deadline',
        'deadline_clock':'continuous monotonic clock including suspend; simulated duration also bounded',
        'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [SOURCE/'nav_control_llm.py',SOURCE/'steering.py',SOURCE/'LLM_SPEC.txt']},
        'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'policy_sha256':hashlib.sha256(POLICY_PATH.read_bytes()).hexdigest(),
        'scene_sha256':hashlib.sha256(scene.read_bytes()).hexdigest(),
        'archive_extraction_active':False,'release_adapter':'public paths; private source supplied externally'}
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    slow={'command':[0.,0.,0.],'frame':-1,'version':0,'result':None,'first_image_request_ns':None}
    active={'command':[0.,0.,0.],'frame':-1,'source':'none'}
    local=None;cloud=None;nav_thread=None;nav_env=None;pending=None;local_results=[]
    warm_result=None;last_local_request_frame=-1
    counters={'tracking_ns':0,'geometry_ns':0,'validity_ns':0,'invalid_ticks':0,'local_requests':0,'rejected_local':0}
    collisions=set();fall=False;completion=False;changed=False;dwelling=0.;frames=0
    robot_bodies={model.body('base').id}
    for i in range(model.nbody):
        if int(model.body_parentid[i]) in robot_bodies:robot_bodies.add(i)
    with (out/'events.jsonl').open('w') as log, mujoco.Renderer(model,height=240,width=320) as renderer,ThreadPoolExecutor(max_workers=1) as pool:
        event_lock=threading.Lock()
        def event(kind,**fields):
            if 'event' in fields:fields['worker_event']=fields.pop('event')
            with event_lock:
                log.write(json.dumps({'event':kind,'monotonic_ns':time.monotonic_ns(),'continuous_ns':continuous_ns(),'utc_ns':time.time_ns(),'sim_s':float(data.time),**fields})+'\n');log.flush()
        try:
            if args.mode in ['frequent','selective']:
                # Keep the API credential out of the local model child process.
                key=os.environ.pop('OPENAI_API_KEY',None)
                try:local=LocalVLM(args.refiner_model,out/'refiner.stderr.log')
                finally:
                    if key is not None:os.environ['OPENAI_API_KEY']=key
                    key=None
                renderer.update_scene(data,camera='front',scene_option=options)
                warm=out/'warmup.jpg';Image.fromarray(renderer.render()).save(warm,quality=70)
                warm_result=local.predict(warm,'warmup',POINT_PROMPT,output_format='point')
                (out/'refiner-startup.json').write_text(json.dumps({'ready':local.ready,'warmup':warm_result},indent=2)+'\n')
            if args.mode!='tracking':cloud=CloudChat(out)
            def chat(body,timeout):
                if sensor.closed:raise EpisodeStopped()
                if any(m.get('images') for m in body['messages']) and slow['first_image_request_ns'] is None:slow['first_image_request_ns']=time.monotonic_ns()
                event('cloud_request',source_snapshot=sensor.planner_snapshot,requested_alias=body.get('model'))
                return cloud(body,timeout)
            def publish(x,y,z):
                if sensor.closed:raise EpisodeStopped()
                slow.update(command=[float(x),float(y),float(z)],frame=sensor.planner_snapshot['frame'],version=slow['version']+1)
                event('slow_guidance',command=slow['command'],source_snapshot=sensor.planner_snapshot)
            def stop(_dry):event('source_stop_requested')
            if cloud:
                nav_env=navigation(sensor,chat,publish,stop,out/'source-navigation')
                def native_navigation():
                    try:slow['result']=nav_env['run']('walk to the blue box','the blue box',dry_run=False)
                    except EpisodeStopped:slow['result']=['episode-stopped',None,None]
                    except BaseException as error:slow['result']=['source-error',type(error).__name__,str(error)]
                nav_thread=threading.Thread(target=native_navigation,daemon=True)
            sim_start=float(data.time);wall_start=continuous_s();utc_start_ns=time.time_ns();awake_start_ns=time.monotonic_ns();next_frame=0.;last_slow=0
            p={'visible':False,'bearing':None};clear={n:0. for n in s.ORDER};last_obs_ns=0
            if nav_thread:nav_thread.start()
            event('episode_start')
            while float(data.time)-sim_start<args.duration and continuous_s()-wall_start<args.duration:
                elapsed=float(data.time)-sim_start
                trigger=elapsed>=.6
                if args.scenario!='static' and not changed and trigger:
                    if args.scenario=='moved':data.mocap_pos[target]=[2.,-.7,.22]
                    else:data.mocap_pos[distractor]=[3.5,2.,1.]
                    mujoco.mj_forward(model,data);changed=True
                    event('scripted_change_oracle',kind_of_change=args.scenario)
                if elapsed+1e-9>=next_frame:
                    acquired=time.monotonic_ns();renderer.update_scene(data,camera='front',scene_option=options)
                    rgb=renderer.render().copy();buffer=io.BytesIO();Image.fromarray(rgb).save(buffer,format='JPEG',quality=70);jpeg=buffer.getvalue()
                    image=out/f'front_{frames:04d}.jpg';image.write_bytes(jpeg)
                    begin=time.thread_time_ns();p=pixels(np.array(Image.open(io.BytesIO(jpeg))),fovy);counters['tracking_ns']+=time.thread_time_ns()-begin
                    begin=time.thread_time_ns();grid,yaw=height_view(model,data);clear=s.scan(grid);counters['geometry_ns']+=time.thread_time_ns()-begin
                    sensor.update({'frame':frames,'grid':grid,'yaw':yaw,'jpeg':jpeg,'image':str(image),'acquired_ns':acquired,'sim_s':float(data.time)})
                    last_obs_ns=acquired
                    event('observation',frame=frames,acquired_ns=acquired,image=str(image),image_sha256=hashlib.sha256(jpeg).hexdigest(),pixels=p,clearances=clear,
                        base_xyz_oracle=data.qpos[:3].tolist(),target_xyz_oracle=data.mocap_pos[target].tolist())
                    frames+=1;next_frame+=.1
                if slow['version']!=last_slow:
                    last_slow=slow['version']
                    if slow['frame']>=active['frame']:active={'command':slow['command'].copy(),'frame':slow['frame'],'source':'cloud'}
                if pending is not None and pending[0].done():
                    future,snapshot=pending;result=future.result();candidate=direction_command(result['side'])
                    accepted=useful(candidate,p)[0] and snapshot['frame']>=active['frame']
                    if accepted:active={'command':candidate,'frame':snapshot['frame'],'source':'local'}
                    else:counters['rejected_local']+=1
                    local_results.append({'source_snapshot':snapshot,'accepted':accepted,**result})
                    event('local_result',source_snapshot=snapshot,accepted=accepted,**result);pending=None
                begin=time.thread_time_ns();valid,reason=useful(active['command'],p);counters['validity_ns']+=time.thread_time_ns()-begin
                close_goal=p['visible'] and abs(p['bearing'])<math.radians(20) and clear['center']<=.48
                if local and pending is None and not close_goal and (args.mode=='frequent' or not valid):
                    obs=sensor.obs
                    if obs is not None and obs['frame']!=last_local_request_frame:
                        snapshot={k:obs[k] for k in ['frame','acquired_ns','sim_s']}
                        pending=(pool.submit(local.predict,Path(obs['image']),counters['local_requests'],POINT_PROMPT,'point'),snapshot)
                        counters['local_requests']+=1;event('local_request',source_snapshot=snapshot,previous_command=slow['command'])
                        last_local_request_frame=obs['frame']
                if args.mode=='tracking':
                    desired=[.3 if p['visible'] and abs(p['bearing'])<.25 else 0.,0.,float(np.clip(1.7*(p['bearing'] or 0.),-.6,.6))]
                else:desired=active['command'].copy() if valid else [0.,0.,0.]
                if not valid:counters['invalid_ticks']+=1
                desired=[float(np.clip(desired[0],0,.35)),0.,float(np.clip(desired[2],-.6,.6))]
                # Shared geometric admission. The same close-goal stop applies
                # to every mode; it never reads the oracle target coordinates.
                if desired[0]>0:
                    cap=s.speed_for(clear['center'])
                    if cap is None:desired=[0.,0.,0.]
                    else:desired[0]=min(desired[0],cap)
                if close_goal or time.monotonic_ns()-last_obs_ns>500_000_000:desired=[0.,0.,0.]
                if gait.step_count%gait.decimation==0:event('executed_velocity',command=desired,source=active['source'],command_valid=valid,validity_reason=reason,close_goal=close_goal)
                gait.update(model,data,desired);mujoco.mj_step(model,data)
                distance=float(np.linalg.norm(data.qpos[:2]-data.mocap_pos[target,:2]));speed=float(np.linalg.norm(data.sensor('local_linvel').data[:2]))
                for contact in data.contact:
                    bodies={int(model.geom_bodyid[contact.geom1]),int(model.geom_bodyid[contact.geom2])}
                    if bodies&robot_bodies:
                        names=[model.geom(contact.geom1).name,model.geom(contact.geom2).name]
                        collisions.update(name for name in environment[1:] if name in names)
                if not healthy(model,data):fall=True;break
                dwelling=dwelling+float(model.opt.timestep) if distance<=.9 and speed<=.08 else 0.
                if dwelling>=.5:completion=True;break
                remaining=wall_start+float(data.time)-sim_start-continuous_s()
                if remaining>0:time.sleep(remaining)
            end_wall=continuous_s();utc_end_ns=time.time_ns();awake_end_ns=time.monotonic_ns();sensor.close();event('episode_motion_ended')
            # Let outstanding requests finish so their usage is included. Their
            # late results cannot reach movement publishing after sensor.close.
            if pending:
                result=pending[0].result(timeout=120)
                local_results.append({'source_snapshot':pending[1],'accepted':False,'completed_after_motion_end':True,**result})
            if nav_thread:nav_thread.join(timeout=190);assert not nav_thread.is_alive(),'Source worker did not stop'
            renderer.update_scene(data,camera='front',scene_option=options);Image.fromarray(renderer.render()).save(out/'front-final.png')
            summary={'status':'completed','parameters':vars(args),'correct_goal_completion':completion and not collisions and not fall,
                'goal_dwell_reached':completion,'fell':fall,'collisions':sorted(collisions),'final_distance_m':distance,
                'elapsed_sim_s':float(data.time)-sim_start,'elapsed_motion_wall_s':end_wall-wall_start,'elapsed_utc_s':(utc_end_ns-utc_start_ns)/1e9,'elapsed_awake_clock_s':(awake_end_ns-awake_start_ns)/1e9,'timing_continuity_valid':abs((utc_end_ns-utc_start_ns)/1e9-(end_wall-wall_start))<1 and abs((awake_end_ns-awake_start_ns)/1e9-(end_wall-wall_start))<1,
                'simulation_real_time_factor':(float(data.time)-sim_start)/(end_wall-wall_start),
                'deadline_clock':'continuous monotonic clock including suspend',
                'drain_wall_s':continuous_s()-end_wall,'local_requests':counters['local_requests'],
                'local_responses':len(local_results),'local_generation_wall_ms_sum':sum(r['inference_wall_ms'] for r in local_results),
                'local_process_cpu_ms_sum':sum(r.get('inference_process_cpu_ms',0) for r in local_results),
                'cloud_requests':len(cloud.calls) if cloud else 0,'cloud_estimated_usd':sum(r.get('estimated_usd',0) for r in cloud.calls) if cloud else 0,
                'cloud_failures':sum(r['state']!='completed' for r in cloud.calls) if cloud else 0,
                'source_result':slow['result'],'frames':frames,'counters':counters,'change_executed':changed,
                'model_response_costs_complete':True,'startup_warmup_recorded_separately':bool(local),
                'local_model_load_wall_ms':local.ready['load_wall_ms'] if local else 0,
                'local_warmup_wall_ms':warm_result['inference_wall_ms'] if warm_result else 0,
                'energy_measured':False,'lab_reproduction':False,'published_refiner_reproduction':False}
            (out/'local-responses.json').write_text(json.dumps(local_results,indent=2)+'\n')
            (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');event('episode_end',summary=summary)
        finally:
            sensor.close()
            if local:local.close()
    print(json.dumps({'output':str(out),**summary},indent=2),flush=True)
    return out,summary

def main():
    parser=argparse.ArgumentParser(description='Original model experiment with externally supplied lab source.');parser.add_argument('--mode',choices=['slow_only','frequent','selective','tracking'],required=True)
    parser.add_argument('--scenario',choices=['static','moved','irrelevant'],default='static')
    parser.add_argument('--duration',type=float,default=30);parser.add_argument('--plan-period',type=float,default=8)
    parser.add_argument('--refiner-model',default='qwen35-4b-4bit')
    parser.add_argument('--phase',default='engineering-v3-calibration');args=parser.parse_args()
    assert 0<args.duration<=60 and args.plan_period>=6
    run(args)

if __name__=='__main__':main()
