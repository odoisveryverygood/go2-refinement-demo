"""Self-contained robot navigation chosen by a real cloud vision-language model.

This new public demonstration is separate from the saved historical comparisons.
No local color tracker chooses directions here. A range-only proximity veto can stop motion.
"""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import io
import json
import os
from pathlib import Path
import sys
import time
import urllib.request

import mujoco
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from demo import setup, reset, healthy, move_target

PROMPT = ('You control a robot dog. Its goal is to approach the blue box. '
          'Look at this CURRENT front-camera image. Choose left or right to face the box, '
          'forward when it is ahead, stop when very close, and stop if uncertain. '
          'Return ONLY JSON: {"direction":"left|right|forward|stop","reason":"brief explanation"}.')
COMMANDS = {'left':[0.,0.,.6], 'right':[0.,0.,-.6],
            'forward':[.3,0.,0.], 'stop':[0.,0.,0.]}


def predict(jpeg, model):
    key = os.environ.get('OPENAI_API_KEY')
    if not key:
        raise RuntimeError('Set your own OPENAI_API_KEY in the process environment')
    payload = {'model':model, 'temperature':0, 'max_completion_tokens':100, 'store':False,
               'response_format':{'type':'json_object'},
               'messages':[{'role':'user','content':[
                   {'type':'text','text':PROMPT},
                   {'type':'image_url','image_url':{'url':'data:image/jpeg;base64,'+
                       base64.b64encode(jpeg).decode(),'detail':'low'}}]}]}
    req = urllib.request.Request('https://api.openai.com/v1/chat/completions',
          data=json.dumps(payload).encode(), headers={
              'Content-Type':'application/json','Authorization':'Bearer '+key})
    started = time.monotonic()
    with urllib.request.urlopen(req, timeout=45) as response:
        result = json.load(response)
    action = json.loads(result['choices'][0]['message']['content'])
    if action.get('direction') not in COMMANDS:
        raise RuntimeError('Model returned an unsupported direction')
    return {**action, 'model':result.get('model'), 'usage':result.get('usage'),
            'latency_s':time.monotonic()-started}


def admitted(model, data, command):
    if command[0] <= 0:
        return command, False
    base = model.body('base').id
    direction = data.xmat[base].reshape(3,3)[:,0].copy()
    direction[2] = 0
    direction /= np.linalg.norm(direction)
    origin = data.xpos[base].copy()+.35*direction
    origin[2] = .25
    hit = np.array([-1], dtype=np.int32)
    distance = mujoco.mj_ray(model, data, origin, direction,
                            np.array([0,0,0,0,0,1],dtype=np.uint8), True, base, hit)
    stop = 0 <= distance < .5
    return ([0.,0.,0.] if stop else command), stop


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', default='gpt-4.1-2025-04-14')
    p.add_argument('--seconds', type=float, default=30)
    p.add_argument('--max-calls', type=int, default=12)
    p.add_argument('--interval', type=float, default=2)
    p.add_argument('--headless', action='store_true')
    p.add_argument('--no-api-smoke', action='store_true', help='Validate simulation only; no model decisions')
    args = p.parse_args()
    if not 0 < args.seconds <= 60 or not 1 <= args.max_calls <= 32 or args.interval < 1:
        p.error('seconds must be (0,60], max-calls 1..32, interval >=1')
    if not args.no_api_smoke and not os.environ.get('OPENAI_API_KEY'):
        p.error('An OPENAI_API_KEY is required; --no-api-smoke only checks plumbing')
    root = Path(__file__).resolve().parents[1]
    out = root/'runs'/('vlm-'+time.strftime('%Y%m%dT%H%M%S'))
    out.mkdir(parents=True, exist_ok=False)
    model, data, gait = setup()
    reset(model, data, gait)
    options = mujoco.MjvOption()
    options.geomgroup[5] = 1
    # setup() puts room geometry in group 5 for ray sensing; overview can see all.
    viewer = None
    if not args.headless:
        from mujoco import viewer as mjviewer
        viewer = mjviewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False)
        with viewer.lock():
            viewer.cam.lookat[:] = [1.,0.,.25]
            viewer.cam.distance = 4.3
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -32
            # Hide walls only from overview; front renderer and rays keep them.
            for name in ['far_wall','left_wall','right_wall']:
                model.geom_group[model.geom(name).id] = 1
            viewer.opt.geomgroup[1] = 0
    pending = None
    command = [0.,0.,0.]
    decision = 'Waiting for VLM' if not args.no_api_smoke else 'NO API: smoke check only'
    records = []
    initial_position = data.qpos[:3].copy().tolist()
    motion_steps = 0
    submitted = 0
    next_query = 0.
    next_camera = 0.
    next_view = 0.
    clock = time.monotonic()
    sim_start = float(data.time)
    rgb = None
    changed = False
    failure = None
    try:
        with mujoco.Renderer(model,height=240,width=320) as renderer, ThreadPoolExecutor(max_workers=1) as pool:
            while time.monotonic()-clock < args.seconds and (viewer is None or viewer.is_running()):
                elapsed = time.monotonic()-clock
                if not changed and elapsed > args.seconds/2:
                    move_target(model,data,-1)
                    changed = True
                if float(data.time) >= next_camera:
                    renderer.update_scene(data,camera='front',scene_option=options)
                    rgb = renderer.render().copy()
                    buf = io.BytesIO()
                    Image.fromarray(rgb).save(buf,format='JPEG',quality=70)
                    jpeg = buf.getvalue()
                    next_camera = float(data.time)+.1
                if pending is not None and pending.done():
                    result = pending.result()
                    records.append(result)
                    command = COMMANDS[result['direction']].copy()
                    decision = f"VLM: {result['direction']} ({result['latency_s']:.1f}s)"
                    print(json.dumps(result), flush=True)
                    pending = None
                    next_query = elapsed+args.interval
                if not args.no_api_smoke and pending is None and elapsed >= next_query and submitted < args.max_calls:
                    Image.fromarray(rgb).save(out/f'request-{submitted:03d}.jpg')
                    pending = pool.submit(predict,jpeg,args.model)
                    submitted += 1
                for name in ['far_wall','left_wall','right_wall']:
                    model.geom_group[model.geom(name).id] = 5
                desired, vetoed = admitted(model,data,command)
                if viewer:
                    for name in ['far_wall','left_wall','right_wall']:
                        model.geom_group[model.geom(name).id] = 1
                if not healthy(model,data):
                    raise RuntimeError('Robot became unhealthy')
                gait.update(model,data,desired)
                mujoco.mj_step(model,data)
                if any(abs(v) > 1e-9 for v in desired):
                    motion_steps += 1
                if viewer and elapsed >= next_view:
                    viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150,
                        mujoco.mjtGridPos.mjGRID_TOPLEFT,
                        'MODEL-CONTROLLED DEMO (new public runner)\n'+decision+
                        f'\nSubmitted calls: {submitted} | proximity veto: {vetoed}', ''))
                    viewer.set_images((mujoco.MjrRect(20,20,320,240),rgb))
                    viewer.sync()
                    next_view = elapsed+1/30
                remaining = clock+float(data.time)-sim_start-time.monotonic()
                if remaining > 0:
                    time.sleep(remaining)
            if pending is not None:
                records.append({**pending.result(), 'completed_after_motion_end':True})
            Image.fromarray(rgb).save(out/'front-final.png')
    except Exception as error:
        failure = type(error).__name__
        raise
    finally:
        if viewer:
            viewer.close()
        (out/'model-responses.json').write_text(json.dumps(records,indent=2)+'\n')
        (out/'summary.json').write_text(json.dumps({'model':args.model,'submitted_calls':submitted,
            'completed_responses':len(records),'simulation_seconds':float(data.time)-sim_start,
            'initial_base_xyz':initial_position,'final_base_xyz':data.qpos[:3].tolist(),
            'model_motion_physics_steps':motion_steps,
            'api_disabled':args.no_api_smoke,'failure_type':failure,
            'scope':'new public model-controlled demo; not historical experiment reproduction'},indent=2)+'\n')
        print(f'Output: {out}',flush=True)


if __name__ == '__main__':
    main()
