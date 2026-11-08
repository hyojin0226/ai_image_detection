"""Check the final release or package its local weights. Never run inference."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'releases/v1.0.0.json'


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def checked_path(relative):
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError('Invalid release path: ' + relative)
    return path


def check(code_only=False):
    lock = json.loads(MANIFEST.with_suffix('.lock.json').read_text(encoding='utf-8'))
    if sha(MANIFEST) != lock['manifest_sha256']:
        raise ValueError('Release manifest changed')
    release = json.loads(MANIFEST.read_text(encoding='utf-8'))
    expected = dict(release['code_sha256'])
    if not code_only:
        expected.update(release['weights_sha256'])
    for name, digest in expected.items():
        path = checked_path(name)
        if not path.is_file():
            raise ValueError('Missing release file: ' + name + '. See README.md for model placement.')
        if sha(path) != digest:
            raise ValueError('Release file changed: ' + name)
    if release['threshold'] != .5 or release['planned_frames'] != 64:
        raise ValueError('Unexpected final model specification')
    return release


def package():
    release = check()
    output = ROOT / 'distribution/final_research_v1.zip'
    output.parent.mkdir(exist_ok=True)
    if output.exists():
        raise ValueError('Preserve the existing package: ' + str(output))
    with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name in release['weights_sha256']:
            path = checked_path(name)
            archive.write(path, str(path.relative_to(ROOT / 'models')))
    print('Model bundle: ' + str(output))
    print('SHA-256: ' + sha(output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['check', 'package'], nargs='?', default='check')
    parser.add_argument('--code-only', action='store_true')
    args = parser.parse_args()
    if args.action == 'package':
        if args.code_only:
            parser.error('Packaging requires the complete model')
        package()
    else:
        release = check(args.code_only)
        print('PASS: ' + release['version'] + (' code' if args.code_only else ' code and model weights'))
        print('No training, downloads or video predictions were run.')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError) as error:
        raise SystemExit(str(error))
