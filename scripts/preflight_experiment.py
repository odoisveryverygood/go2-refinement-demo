"""Check prerequisites without API calls, downloads, or exposing credentials."""
import importlib.util
import os
from pathlib import Path
import platform
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]

def main():
    source = Path(os.environ.get('ROBOTICS_LAB_SOURCE',str(ROOT/'private-lab-source'))).expanduser()
    checks = {
        'simulation dependencies':all(importlib.util.find_spec(x) for x in ['mujoco','numpy','PIL','onnxruntime']),
        'private lab source supplied':all((source/n).is_file() for n in ['nav_control_llm.py','steering.py','LLM_SPEC.txt']),
        'cloud key present (not validated)':bool(os.environ.get('OPENAI_API_KEY')),
        '8 GiB free reserve':shutil.disk_usage(ROOT).free > 8*1024**3,
        'Apple Silicon for original MLX local backend':sys.platform=='darwin' and platform.machine()=='arm64',
        'local model files present':(ROOT/'assets/models/qwen35-4b-4bit/model.safetensors').is_file(),
    }
    for label, passed in checks.items():
        print(('OK   ' if passed else 'NEED ')+label)
    return 0 if all(checks.values()) else 1

if __name__ == '__main__':
    raise SystemExit(main())
