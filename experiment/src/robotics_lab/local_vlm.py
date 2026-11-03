"""One private subprocess per local VLM, usable from a planner worker thread."""
import json
import os
import sys
from pathlib import Path
import select
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]


class LocalVLM:
    def __init__(self, model_name: str, log_path: Path):
        self.log = log_path.open("w")
        self.process = subprocess.Popen(
            [os.environ.get('ROBOTICS_VLM_PYTHON',str(ROOT/'.vlm-venv/bin/python') if (ROOT/'.vlm-venv/bin/python').exists() else sys.executable), str(ROOT / "scripts/vlm_worker.py"),
             "--model", str(ROOT / "assets/models" / model_name)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
            text=True, bufsize=1,
            env={**__import__("os").environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
        try:
            self.ready = self._read(120)
        except Exception:
            self.close()
            raise
        if self.ready.get("event") != "ready":
            self.close()
            raise RuntimeError(f"Worker did not load: {self.ready}")

    def _read(self, timeout):
        available, _, _ = select.select([self.process.stdout], [], [], timeout)
        if not available:
            raise TimeoutError("VLM worker timed out; inspect its stderr log.")
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("VLM worker exited; inspect its stderr log.")
        return json.loads(line)

    def predict(self, image: Path, request_id=0, prompt=None, output_format='direction') -> dict:
        start = time.monotonic_ns()
        request = {"image": str(image.resolve()), "request_id": request_id}
        request['output_format']=output_format
        if prompt is not None:
            request["prompt"] = prompt
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        result = self._read(90)
        result["roundtrip_wall_ms"] = (time.monotonic_ns() - start) / 1e6
        return result

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.log.close()
