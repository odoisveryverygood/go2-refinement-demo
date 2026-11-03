"""Load reviewed pure source definitions while excluding lab startup and ROS.

The supplied source remains untouched. Sensor/transport/parser substitutions
are explicit. Aligned 16x32 views satisfy steering.py's documented interface.
"""
import ast
import base64
import json
import math
import os
from pathlib import Path
import time
from types import SimpleNamespace
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
SOURCE=Path(os.environ.get('ROBOTICS_LAB_SOURCE',str(ROOT/'private-lab-source'))).expanduser().resolve()

class EpisodeStopped(BaseException):
    pass

def definitions(path,env,names=None,assignments=False):
    nodes=[]
    for node in ast.parse(path.read_text()).body:
        if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and (names is None or node.name in names):nodes.append(node)
        elif assignments and isinstance(node,ast.Assign):nodes.append(node)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),env)
    return env

def geometry():
    # These are documented source conventions, not an invented implementation
    # of the missing world-map conversion helper. Only aligned views are used.
    constants=SimpleNamespace(RES_32=.24,NO_DATA_B=-128,NO_DATA_F=1e9,
        DROP_CM=-10,TALL_CM=20,DEPTH_CELLS=16,WIDTH_CELLS=32)
    env=definitions(SOURCE/'steering.py',{'np':np,'math':math,'hr':constants},assignments=True)
    return SimpleNamespace(**env)

def provisional_parse(raw):
    """Adapter JSON validation, to be replaced/checked against recovered V.parse."""
    try:
        value=json.loads(raw)
        if value['status'] not in ['continue','done','failed']:raise ValueError('status')
        x,y,z=(float(value.get(k,0)) for k in ['x','y','z'])
        if not all(math.isfinite(v) for v in [x,y,z]):raise ValueError('nonfinite')
        if abs(y)>1e-9:raise ValueError('sideways motion')
        return {'status':value['status'],'x':float(np.clip(x,0,.6)),'y':0.,
                'z':float(np.clip(z,-1,1)),'why':str(value.get('why',''))[:160]},None
    except Exception:return None,'invalid JSON command'

def navigation(sensor,chat,publish,stop,out):
    s=geometry();tree=ast.parse((SOURCE/'nav_control_llm.py').read_text())
    env={'np':np,'math':math,'os':os,'time':time,'json':json,'base64':base64,'S':s,
         'SensorReceiver':lambda **_kwargs:sensor,'RX_HOST':'simulation-only','RX_PORT':0,
         'LOG_DIR':str(out),'LLM_SYSTEM':(SOURCE/'LLM_SPEC.txt').read_text(),'MAX_LOOKS':0,
         '_chat':chat,'V':SimpleNamespace(parse=provisional_parse,
              wait_for_fresh_map=sensor.wait_map,publish_velocity=publish,_stop_robot=stop)}
    literals={'STEP_BUDGET','SETTLE_S','APPROACH_M','ARRIVE_M','CREEP_X','LATCH_MAX_STEPS',
              'CAM_BEARING','MEMORY_STEPS','SIDE_DEADBAND','VLM_SYSTEM','VLM_MODEL','LLM_MODEL','KEEP_ALIVE'}
    for node in tree.body:
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id in literals:
            env[node.targets[0].id]=ast.literal_eval(node.value)
    names={'ask_vlm','ask_vlm_question','ask_llm','sector_of','vet','grid_and_yaw','creep_action','run'}
    definitions(SOURCE/'nav_control_llm.py',env,names)
    return env

def refiner_prompt(previous,condition_on_previous=False,spatial=False):
    if spatial:
        return ('Find the blue box. Classify its horizontal position in the whole image. '
                'CENTER means its center lies in the middle 20 percent of the image width. '
                'Otherwise use LEFT or RIGHT of that central band. '
                'If no blue box is visible use ABSENT. Reply with exactly one word: LEFT, CENTER, RIGHT, or ABSENT.')
    if not condition_on_previous:
        # Previously frozen forced-choice calibration prompt. This is fresh
        # direction replacement, not a command-conditioned trained actor.
        return 'Locate the blue box in the image. Reply with one word: LEFT, CENTER, RIGHT, or ABSENT.'
    return ('Locate the blue box in the current image. The previous movement '
            f'({previous}) may be outdated. Reply with the CURRENT location only: '
            'LEFT, CENTER, RIGHT, or ABSENT.')
