"""Import the three required files from your own collaborator ZIP, locally only."""
import argparse
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = {'nav_control_llm.py', 'steering.py', 'LLM_SPEC.txt'}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('zip_path', type=Path)
    args = p.parse_args()
    destination = ROOT/'private-lab-source'
    with zipfile.ZipFile(args.zip_path) as archive:
        selected = {}
        for name in REQUIRED:
            matches = [i for i in archive.infolist() if Path(i.filename).name == name
                       and not i.is_dir() and '__MACOSX/' not in i.filename]
            if len(matches) != 1 or matches[0].file_size > 2_000_000:
                raise SystemExit(f'Expected exactly one reasonably sized {name}')
            selected[name] = archive.read(matches[0])
        if destination.exists() and any((destination/n).exists() for n in REQUIRED):
            raise SystemExit('Destination already contains lab files; no files overwritten')
        destination.mkdir(exist_ok=True)
        for name, content in selected.items():
            (destination/name).write_bytes(content)
    print('Imported three files into ignored private-lab-source/. Do not commit them.')

if __name__ == '__main__':
    main()
