"""Build one offline Windows executable from existing, verified installed packages.

Never copies datasets, account settings, caches or unrelated site-packages.
Run with the same CPython 3.12 x64 used by the final model.
"""
import argparse
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '.tools/python'))
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


def build(node, redist, reuse_payload=False):
    if sys.version_info[:2] != (3, 12) or sys.maxsize < 2**32:
        raise RuntimeError('Use CPython 3.12 x64')
    sys.path.insert(0, str(ROOT / 'tools'))
    from release import check
    release = check()
    work = ROOT / '.local/windows_desktop_build'
    work.mkdir(parents=True, exist_ok=True)
    if not NODE_LICENSE.exists():
        version = subprocess.check_output([str(node), '--version'], text=True).strip()
        if not version.startswith('v') or any(c not in 'v0123456789.' for c in version):
            raise ValueError('Unexpected Node.js version')
        with urllib.request.urlopen('https://raw.githubusercontent.com/nodejs/node/' + version + '/LICENSE', timeout=45) as response:
            license_text = response.read()
        if b'Permission is hereby granted' not in license_text:
            raise ValueError('Node.js license was not retrieved')
        NODE_LICENSE.write_bytes(license_text)
    payload = work / 'payload.zip'
    roots = ['tensorflow', 'keras', 'numpy', 'Pillow', 'imageio-ffmpeg', 'yt-dlp', 'yt-dlp-ejs']
    distributions = {}
    pending = roots[:]
    while pending:
        name = pending.pop()
        key = canonicalize_name(name)
        if key in distributions:
            continue
        dist = metadata.distribution(name)
        distributions[key] = dist
        for text in dist.requires or []:
            requirement = Requirement(text)
            if requirement.marker is None or requirement.marker.evaluate({'extra': ''}):
                pending.append(requirement.name)
    files = {}
    def add(source, name):
        source = Path(source)
        if source.is_file() and '__pycache__' not in source.parts and source.suffix != '.pyc':
            if name in files and files[name] != source:
                if files[name].read_bytes() != source.read_bytes():
                    raise ValueError('Conflicting runtime package: ' + name)
            files[name] = source
    home = Path(sys.base_prefix)
    for name in ('python.exe', 'pythonw.exe', 'python3.dll', 'python312.dll', 'vcruntime140.dll', 'vcruntime140_1.dll', 'LICENSE.txt'):
        add(home / name, 'runtime/python/' + name)
    for directory in ('DLLs', 'tcl', 'Lib'):
        for source in (home / directory).rglob('*'):
            relative = source.relative_to(home)
            if 'site-packages' not in relative.parts and not any(p in ('test', 'tests', 'idlelib', 'ensurepip') for p in relative.parts):
                add(source, 'runtime/python/' + relative.as_posix())
    for dist in distributions.values():
        for entry in dist.files or []:
            if '..' in entry.parts or entry.suffix == '.pyc':
                continue
            # Training C/C++ headers are not needed to run the installed wheel.
            if len(entry.parts) > 1 and entry.parts[:2] == ('tensorflow', 'include'):
                continue
            add(dist.locate_file(entry), 'runtime/python/Lib/site-packages/' + entry.as_posix())
    for source in redist.glob('*.dll'):
        # Use Microsoft's redistribution directory, never DLLs copied from System32.
        files['runtime/python/' + source.name] = source
    add(node, 'runtime/node/node.exe')
    for name in release['code_sha256']:
        add(ROOT / name, name)
    for name in release['weights_sha256']:
        add(ROOT / name, name)
    for name in ('releases/v1.0.0.json', 'releases/v1.0.0.lock.json', 'LICENSE', 'desktop/launch.py'):
        add(ROOT / name, name)
    package = {'version': '1.0.0-desktop', 'platform': 'Windows x64',
               'python': sys.version, 'model_version': release['version'],
               'dependencies': {key: dist.version for key, dist in sorted(distributions.items())},
               'code_sha256': release['code_sha256'], 'weights_sha256': release['weights_sha256'],
               'training_or_test_evaluation': False}
    if reuse_payload:
        previous = json.loads((work / 'build.json').read_text(encoding='utf-8'))
        with payload.open('rb') as stream:
            assert hashlib.file_digest(stream, 'sha256').hexdigest() == previous['payload_sha256']
        with zipfile.ZipFile(payload) as archive:
            for name in [*release['code_sha256'], *release['weights_sha256'], 'desktop/launch.py']:
                assert archive.read(name) == (ROOT / name).read_bytes(), name
    else:
        with zipfile.ZipFile(payload, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            total = len(files)
            for index, (name, source) in enumerate(sorted(files.items()), 1):
                archive.write(source, name)
                if index % 500 == 0 or index == total:
                    print(f'PACKAGE {index}/{total}', flush=True)
            archive.writestr('desktop/package.json', json.dumps(package, ensure_ascii=False, indent=2))
            archive.writestr('runtime/node/LICENSE.txt', NODE_LICENSE.read_text(encoding='utf-8'))
            archive.writestr('THIRD_PARTY.txt', 'Includes CPython (PSF), TensorFlow (Apache-2.0), Keras (Apache-2.0), NumPy (BSD), Pillow (HPND), imageio-ffmpeg (BSD), FFmpeg (GPL build), yt-dlp (Unlicense), yt-dlp-ejs (Unlicense), Node.js (MIT and bundled dependencies), and Microsoft Visual C++ Runtime (redistributable).\nSee runtime package metadata/license files. FFmpeg build/source: https://github.com/imageio/imageio-binaries and https://www.gyan.dev/ffmpeg/builds/ .\nMicrosoft runtime license: https://visualstudio.microsoft.com/license-terms/vs2022-cruntime/ .\n')
    digest = hashlib.file_digest(payload.open('rb'), 'sha256').hexdigest()
    parts_dir = ROOT / 'desktop/runtime'
    parts_dir.mkdir(exist_ok=True)
    part_size = 48 * 1024**2
    parts = []
    with payload.open('rb') as stream:
        while block := stream.read(part_size):
            name = 'runtime.' + str(len(parts) + 1).zfill(3)
            path = parts_dir / name
            path.write_bytes(block)
            parts.append({'file': 'desktop/runtime/' + name, 'bytes': len(block), 'sha256': hashlib.sha256(block).hexdigest()})
    source = (ROOT / 'desktop/Launcher.cs').read_text(encoding='utf-8').replace('__PAYLOAD_SHA256__', digest).replace('__PACKAGE_ID__', 'v1.0.0-' + digest[:12]).replace('__PART_COUNT__', str(len(parts)))
    generated = work / 'Launcher.generated.cs'
    generated.write_text(source, encoding='utf-8-sig')
    output = ROOT / 'distribution/AI-Video-Classifier-Windows-x64.exe'
    launcher = ROOT / '실행.exe'
    compiler = Path(os.environ.get('WINDIR', r'C:\Windows')) / 'Microsoft.NET/Framework64/v4.0.30319/csc.exe'
    subprocess.run([str(compiler), '/nologo', '/target:winexe', '/platform:x64', '/optimize+',
                    '/reference:System.Windows.Forms.dll', '/reference:System.Drawing.dll',
                    '/reference:System.IO.Compression.dll', '/reference:System.IO.Compression.FileSystem.dll',
                    '/out:' + str(launcher), str(generated)], check=True)
    shutil.copyfile(launcher, output)
    offset = output.stat().st_size
    with output.open('ab') as target, payload.open('rb') as stream:
        shutil.copyfileobj(stream, target)
        target.write(struct.pack('<qq8s', offset, payload.stat().st_size, b'AIVIDEO1'))
    executable_hash = hashlib.file_digest(output.open('rb'), 'sha256').hexdigest()
    details = {'file': output.name, 'bytes': output.stat().st_size, 'sha256': executable_hash,
               'payload_sha256': digest, 'bundled_files': len(files), 'package_id': 'v1.0.0-' + digest[:12],
               'git_launcher': launcher.name, 'parts': parts}
    (work / 'build.json').write_text(json.dumps(details, indent=2), encoding='utf-8')
    print(json.dumps(details, indent=2), flush=True)


NODE_LICENSE = ROOT / '.local/windows_desktop_build/node-LICENSE.txt'
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--node', type=Path, required=True)
    parser.add_argument('--redist', type=Path, required=True)
    parser.add_argument('--reuse-payload', action='store_true', help='Reuse only a hash-verified unchanged payload')
    args = parser.parse_args()
    build(args.node.resolve(), args.redist.resolve(), args.reuse_payload)
