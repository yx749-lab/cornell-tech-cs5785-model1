"""Verify all packaged file hashes without third-party dependencies."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]

def main():
    manifest = json.loads((ROOT / 'MANIFEST.json').read_text())
    for item in manifest['files']:
        relative = Path(item['path'])
        assert not relative.is_absolute() and '..' not in relative.parts
        path = ROOT / relative
        assert path.is_file(), f'Missing: {relative}'
        assert path.stat().st_size == item['bytes'], f'Size mismatch: {relative}'
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256'], f'Hash mismatch: {relative}'
    print(f'PASS: {len(manifest["files"])} packaged files match their SHA-256 manifest.')

if __name__ == '__main__':
    main()
