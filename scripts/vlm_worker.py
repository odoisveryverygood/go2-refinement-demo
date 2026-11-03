"""Private local JSON-lines worker for real MLX VLM inference. No network service.

Only explicit local model weights and image paths are accepted. Model downloads
are a separate setup step. Stdout is reserved for the protocol; logs use stderr.
"""
import argparse
import contextlib
import json
from pathlib import Path
import re
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    model_path = Path(args.model).resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"Local weights missing: {model_path}")
    started = time.monotonic_ns()
    with contextlib.redirect_stdout(sys.stderr):
        import mlx.core as mx
        from mlx_vlm import load, generate
        from mlx_vlm.prompt_utils import apply_chat_template
        from mlx_vlm.utils import load_config
        if not mx.metal.is_available():
            raise RuntimeError("This worker requires the Mac's Metal backend.")
        model, processor = load(str(model_path))
        config = load_config(str(model_path))
    template_kwargs={'enable_thinking':False} if config.get('model_type')=='qwen3_5' else {}
    print(json.dumps({"event": "ready", "load_wall_ms": (time.monotonic_ns()-started)/1e6,
                      "model_path": str(model_path), "metal": True,
                      "template_kwargs":template_kwargs}), flush=True)
    for line in sys.stdin:
        request = json.loads(line)
        image_path = Path(request["image"]).resolve()
        prompt = request.get("prompt", "Locate the blue box in the image. Reply with one word: LEFT, CENTER, RIGHT, or ABSENT.")
        output_format=request.get('output_format','direction')
        assert output_format in ['direction','point']
        inference_start = time.monotonic_ns()
        cpu_start = time.process_time_ns()
        with contextlib.redirect_stdout(sys.stderr):
            formatted = apply_chat_template(processor, config, prompt, num_images=1,**template_kwargs)
            output = generate(model, processor, formatted, [str(image_path)],
                              max_tokens=48 if output_format=='point' else 12, temperature=0.0, verbose=False)
            mx.synchronize()
        ended = time.monotonic_ns()
        text = output.text if hasattr(output, "text") else str(output)
        point=None
        if output_format=='direction':
            labels = set(re.findall(r"\b(LEFT|CENTER|RIGHT|ABSENT)\b", text.upper()))
            side = next(iter(labels)).lower() if len(labels) == 1 else "unknown"
        else:
            side='unknown'
            try:
                match=re.search(r'\{[^{}]*\}',text)
                parsed=json.loads(match.group())
                point=parsed['point']
                if point is None:side='absent'
                elif (isinstance(point,list) and len(point)==2 and
                      all(type(v) in [int,float] and 0<=v<=1000 for v in point)):
                    side='left' if point[0]<400 else 'right' if point[0]>600 else 'center'
                else:point=None
            except (ValueError,KeyError,AttributeError,TypeError):pass
        result = {"event": "prediction", "request_id": request.get("request_id"),
                  "side": side, "raw_text": text,
                  "output_format":output_format,"normalized_point":point,
                  "inference_start_ns": inference_start, "inference_end_ns": ended,
                  "inference_wall_ms": (ended-inference_start)/1e6,
                  "inference_process_cpu_ms": (time.process_time_ns()-cpu_start)/1e6,
                  "peak_gpu_memory_bytes": int(mx.get_peak_memory()),
                  "generation_tokens": getattr(output, "generation_tokens", None)}
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
