"""Download and verify the pinned 4-bit local model. Requires ~3.1 GB plus reserve."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]

def main():
    from huggingface_hub import snapshot_download
    manifest = json.loads((ROOT/'assets/qwen35-4b-manifest.json').read_text())
    target = ROOT/'assets/models/qwen35-4b-4bit'
    required = sum(f['bytes'] for f in manifest['selected_files']) + 8*1024**3
    if shutil.disk_usage(ROOT).free < required:
        raise SystemExit('Need about 12 GB free for model download and experiment reserve')
    snapshot_download(repo_id=manifest['repo'], revision=manifest['revision'],
                      allow_patterns=[f['name'] for f in manifest['selected_files']],
                      local_dir=str(target))
    for name, expected in manifest['files'].items():
        path = target/name
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(1024*1024), b''):
                digest.update(block)
        if path.stat().st_size != expected['bytes'] or digest.hexdigest() != expected['sha256']:
            raise SystemExit(f'Model verification failed: {name}')
    print('Pinned model downloaded and verified. Original conversion license is unspecified; weights are not redistributed here.')

if __name__ == '__main__':
    main()
