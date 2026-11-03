"""Independent camera-tracking illustration. No language model or private lab code.

This public demo is different from the historical model-comparison runner.
It uses blue pixels for bearing and a synthetic forward ray for proximity.
"""
import argparse
import math
from pathlib import Path
import queue
import sys
import time

import mujoco
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from locomotion import ROOT, Go2Locomotion, build_scene, healthy


def bearing(rgb, fovy):
    p = rgb.astype(np.int16)
    _, xs = np.where((p[:, :, 2] > 90) & (p[:, :, 2] > 2*p[:, :, 0])
                     & (p[:, :, 2] > 2*p[:, :, 1]))
    if len(xs) < 10:
        return None
    normalized = (rgb.shape[1]/2 - float(xs.mean()))/(rgb.shape[1]/2)
    return math.atan(normalized * rgb.shape[1]/rgb.shape[0]
                     * math.tan(math.radians(fovy/2)))


def control(model, data, rgb):
    angle = bearing(rgb, float(model.cam_fovy[model.camera('front').id]))
    if angle is None:
        return [0., 0., 0.], 'Target not visible: stopped'
    if abs(angle) > math.radians(8):
        return [0., 0., float(np.clip(1.7*angle, -.6, .6))], 'Turning toward blue pixels'
    base = model.body('base').id
    rotation = data.xmat[base].reshape(3, 3)
    direction = rotation[:, 0].copy()
    direction[2] = 0
    direction /= np.linalg.norm(direction)
    origin = data.xpos[base].copy() + .35*direction
    origin[2] = .25
    hit = np.array([-1], dtype=np.int32)
    distance = mujoco.mj_ray(model, data, origin, direction,
                            np.array([0, 0, 0, 0, 0, 1], dtype=np.uint8),
                            True, base, hit)
    if 0 <= distance < .50:
        return [0., 0., 0.], 'Proximity stop (illustration, not research success)'
    return [.30, 0., 0.], 'Walking toward blue pixels'


def setup():
    out = ROOT/'runs/public-demo'
    out.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(build_scene(out)))
    data = mujoco.MjData(model)
    gait = Go2Locomotion(model)
    for name in ['floor', 'far_wall', 'left_wall', 'right_wall', 'blue_target']:
        model.geom_group[model.geom(name).id] = 5
    return model, data, gait


def reset(model, data, gait):
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    gait.last_action[:] = 0
    gait.step_count = 0
    for _ in range(250):
        gait.update(model, data, [0., 0., 0.])
        mujoco.mj_step(model, data)
    move_target(model, data, 1)


def move_target(model, data, side):
    target = int(model.body_mocapid[model.body('target').id])
    data.mocap_pos[target] = [2., .7*side, .22]
    mujoco.mj_forward(model, data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--seconds', type=float, default=8)
    parser.add_argument('--record', type=Path)
    parser.add_argument('--snapshot', type=Path)
    args = parser.parse_args()
    if args.record and not args.headless:
        parser.error('--record requires --headless')
    if args.seconds <= 0:
        parser.error('--seconds must be positive')
    model, data, gait = setup()
    reset(model, data, gait)
    options = mujoco.MjvOption()
    options.geomgroup[5] = 1
    camera_options = mujoco.MjvOption()
    camera_options.geomgroup[5] = 1
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [1., 0., .25]
    camera.distance = 4.3
    camera.azimuth = 135
    camera.elevation = -32
    # Hide room walls in overview only; synthetic camera/ray still sees them.
    for name in ['far_wall', 'left_wall', 'right_wall']:
        model.geom_group[model.geom(name).id] = 1
    options.geomgroup[1] = 0
    keys = queue.SimpleQueue()
    viewer = None
    if not args.headless:
        from mujoco import viewer as mjviewer
        viewer = mjviewer.launch_passive(model, data, key_callback=keys.put,
                                              show_left_ui=False, show_right_ui=False)
        with viewer.lock():
            viewer.cam.lookat[:] = camera.lookat
            viewer.cam.distance = camera.distance
            viewer.cam.azimuth = camera.azimuth
            viewer.cam.elevation = camera.elevation
            viewer.opt.geomgroup[1] = 0
            viewer.opt.geomgroup[5] = 1
    paused = not args.headless
    command = [0., 0., 0.]
    side = 1
    next_camera = 0.
    next_view = 0.
    started = float(data.time)
    frames = []
    ever_unhealthy = False
    print('Go2 public camera-tracking demo. SPACE start/pause; M move box; R reset.', flush=True)
    try:
        with mujoco.Renderer(model, height=360, width=640) as renderer:
            while viewer.is_running() if viewer else float(data.time)-started < args.seconds:
                tick = time.monotonic()
                while not keys.empty():
                    key = keys.get()
                    if key == 32:
                        paused = not paused
                    elif key in [77, 109]:
                        side *= -1
                        move_target(model, data, side)
                        next_camera = 0.
                    elif key in [82, 114]:
                        reset(model, data, gait)
                        started = float(data.time)
                        next_camera = 0.
                        paused = True
                elapsed = float(data.time)-started
                if args.headless and elapsed >= args.seconds/2 and side == 1:
                    side = -1
                    move_target(model, data, side)
                if float(data.time) >= next_camera:
                    renderer.update_scene(data, camera='front', scene_option=camera_options)
                    rgb = renderer.render().copy()
                    for name in ['far_wall', 'left_wall', 'right_wall']:
                        model.geom_group[model.geom(name).id] = 5
                    try:
                        command, state = control(model, data, rgb)
                    finally:
                        for name in ['far_wall', 'left_wall', 'right_wall']:
                            model.geom_group[model.geom(name).id] = 1
                    next_camera = float(data.time)+.1
                    if args.headless and (args.record or args.snapshot):
                        renderer.update_scene(data, camera=camera, scene_option=options)
                        panel = Image.fromarray(renderer.render().copy())
                        panel.paste(Image.fromarray(rgb).resize((213, 120)), (12, 228))
                        draw = ImageDraw.Draw(panel)
                        draw.rectangle((0, 0, 640, 42), fill='#17232f')
                        draw.text((12, 8), 'LIVE SIMULATION CAPTURE | CAMERA TRACKER | NO LLM', fill='white')
                        draw.text((12, 25), f'{elapsed:.1f}s   {state}', fill='white')
                        frames.append(panel)
                if not healthy(model, data):
                    ever_unhealthy = True
                    command = [0., 0., 0.]
                    if viewer:
                        paused = True
                    else:
                        raise RuntimeError('Robot became unhealthy during smoke run')
                if not paused:
                    gait.update(model, data, command)
                    mujoco.mj_step(model, data)
                if viewer and tick >= next_view:
                    viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150,
                                      mujoco.mjtGridPos.mjGRID_TOPLEFT,
                                      'Go2 public demo - camera tracker, no LLM\n'
                                      'SPACE start/pause | M move target | R reset\n'
                                      + ('PAUSED' if paused else state), ''))
                    viewer.set_images((mujoco.MjrRect(20, 20, 320, 180),
                                       np.asarray(Image.fromarray(rgb).resize((320, 180)))))
                    viewer.sync()
                    next_view = tick+1/30
                if viewer:
                    remaining = (1/30 if paused else float(model.opt.timestep))-(time.monotonic()-tick)
                    if remaining > 0:
                        time.sleep(remaining)
        if args.snapshot and frames:
            args.snapshot.parent.mkdir(parents=True, exist_ok=True)
            frames[len(frames)//3].save(args.snapshot)
        if args.record and frames:
            args.record.parent.mkdir(parents=True, exist_ok=True)
            frames[0].save(args.record, save_all=True, append_images=frames[1:],
                           duration=100, loop=0, optimize=True)
        print(f'Finished: {float(data.time)-started:.3f} simulated seconds; '
              f'{gait.calls} locomotion calls; ever_unhealthy={ever_unhealthy}; model_calls=0')
    finally:
        if viewer:
            viewer.close()


if __name__ == '__main__':
    main()
