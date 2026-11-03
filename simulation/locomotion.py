"""CPU ONNX Go2 locomotion adapter, separate from any semantic action refiner.

Observation construction and action scaling adapted from discoverse-dev/
MuJoCo-LiDAR, examples/unitree_go2.py (MIT, Copyright 2025 Yufei Jia).
The matching model, policy, reference source, and license are retained locally.
"""
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "assets/vendor/mujoco_lidar"
POLICY_PATH = VENDOR / "examples/onnx/go2_policy.onnx"
HOME = np.array([0.1, 0.9, -1.8, -0.1, 0.9, -1.8,
                 0.1, 0.9, -1.8, -0.1, 0.9, -1.8], dtype=np.float32)


def build_scene(out: Path) -> Path:
    source = VENDOR / "models/unitree_go2/go2_mjx.xml"
    tree = ET.parse(source)
    root = tree.getroot()
    for mesh in root.findall("asset/mesh"):
        mesh.set("file", str((source.parent / mesh.get("file")).resolve()))
    world = root.find("worldbody")
    ET.SubElement(world, "light", pos="1 0 3", dir="0 0 -1", directional="true")
    ET.SubElement(world, "geom", name="floor", type="plane", size="10 10 0.1",
                  rgba="0.65 0.65 0.6 1", friction="0.8 0.02 0.01", group="2")
    for name, pos, size in [("far_wall", "6 0 1", "0.05 3 1"),
                             ("left_wall", "2 3 1", "4 0.05 1"),
                             ("right_wall", "2 -3 1", "4 0.05 1")]:
        ET.SubElement(world, "geom", name=name, type="box", pos=pos, size=size,
                      rgba="0.55 0.55 0.52 1", group="1")
    target = ET.SubElement(world, "body", name="target", mocap="true", pos="3 0.8 0.22")
    ET.SubElement(target, "geom", name="blue_target", type="box", size="0.15 0.15 0.22",
                  rgba="0.02 0.08 0.95 1", group="2")
    camera = world.find("body[@name='base']/camera[@name='head_camera']")
    camera.set("name", "front")
    camera.set("fovy", "75")
    camera.set("pos", "0.32 0 0.10")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="640", offheight="480")
    ET.SubElement(visual, "headlight", ambient="0.4 0.4 0.4", diffuse="0.6 0.6 0.6")
    keys = ET.SubElement(root, "keyframe")
    home = " ".join(map(str, HOME.tolist()))
    ET.SubElement(keys, "key", name="home", qpos="0 0 0.278 1 0 0 0 " + home, ctrl=home)
    path = out / "scene.xml"
    tree.write(path, encoding="unicode")
    return path


class Go2Locomotion:
    """48 observations -> 12 joint targets at 50 Hz; physics at 250 Hz."""
    def __init__(self, model):
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(POLICY_PATH), sess_options=options,
                                            providers=["CPUExecutionProvider"])
        if self.session.get_inputs()[0].shape[-1] != 48:
            raise ValueError("Unexpected policy input: expected the published 48-observation model.")
        self.last_action = np.zeros(12, dtype=np.float32)
        self.calls = 0
        self.wall_ns = 0
        self.cpu_ns = 0
        self.step_count = 0
        self.decimation = round(0.02 / model.opt.timestep)
        names = [model.joint(int(j)).name for j in model.actuator_trnid[:, 0]]
        expected = [f"{leg}_{part}_joint" for leg in ["FL", "FR", "RL", "RR"]
                    for part in ["hip", "thigh", "calf"]]
        if names != expected:
            raise ValueError(f"Joint order differs from the published model: {names}")

    def update(self, model, data, command):
        if self.step_count % self.decimation == 0:
            start = time.perf_counter_ns()
            cpu_start = time.thread_time_ns()
            imu = model.site("imu").id
            rotation = data.site_xmat[imu].reshape(3, 3)
            obs = np.concatenate([
                data.sensor("local_linvel").data,
                data.sensor("gyro").data,
                rotation.T @ np.array([0, 0, -1]),
                data.qpos[7:19] - HOME,
                data.qvel[6:18],
                self.last_action,
                np.asarray(command),
            ]).astype(np.float32)[None, :]
            action = self.session.run(["continuous_actions"], {"obs": obs})[0][0]
            if action.shape != (12,) or not np.isfinite(action).all():
                raise RuntimeError("Invalid locomotion action.")
            self.last_action = action.copy()
            data.ctrl[:] = action * 0.5 + HOME
            self.calls += 1
            self.wall_ns += time.perf_counter_ns() - start
            self.cpu_ns += time.thread_time_ns() - cpu_start
        self.step_count += 1


def healthy(model, data) -> bool:
    rotation = data.xmat[model.body("base").id].reshape(3, 3)
    return bool(np.isfinite(data.qpos).all() and data.qpos[2] > 0.16 and rotation[2, 2] > 0.5)
