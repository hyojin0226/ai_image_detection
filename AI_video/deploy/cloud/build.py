"""Build a small cloud adapter around the pinned, unchanged public release."""
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import urllib.request

BASE=Path(__file__).resolve().parent
FROZEN=BASE/'frozen'
COMMIT='281e2a957c0c5adb590884e82d164f40140d8acf'
RAW=f'https://raw.githubusercontent.com/hyojin0226/ai_image_detection/{COMMIT}/AI_video/'

def fetch(url):
    with urllib.request.urlopen(url,timeout=90) as response:return response.read()

def main():
    FROZEN.mkdir(exist_ok=True)
    manifest=FROZEN/'releases/v1.0.0.json'
    if not manifest.exists():
        manifest.parent.mkdir(parents=True,exist_ok=True)
        manifest.write_bytes(fetch(RAW+'releases/v1.0.0.json'))
    release=json.loads(manifest.read_text())
    files={**release['code_sha256'],**release['weights_sha256']}
    for name,digest in files.items():
        destination=(FROZEN/name).resolve()
        if not destination.is_relative_to(FROZEN.resolve()):raise ValueError('Invalid release path')
        if not destination.exists():
            destination.parent.mkdir(parents=True,exist_ok=True)
            destination.write_bytes(fetch(RAW+name))
        if hashlib.sha256(destination.read_bytes()).hexdigest()!=digest:raise ValueError('Frozen file changed: '+name)
    for name in ['releases/v1.0.0.lock.json']:
        target=FROZEN/name
        if not target.exists():target.write_bytes(fetch(RAW+name))
    lock=json.loads((FROZEN/'releases/v1.0.0.lock.json').read_text())
    if hashlib.sha256(manifest.read_bytes()).hexdigest()!=lock['manifest_sha256']:
        raise ValueError('Frozen release manifest changed')
    # A Linux Node runtime is used only by the normal public Shorts extractor.
    target=BASE/'vendor/node';target.parent.mkdir(exist_ok=True)
    if not target.exists():
        filename='node-v24.22.1-linux-x64.tar.xz'
        url='https://nodejs.org/dist/v24.22.1/'
        sums=fetch(url+'SHASUMS256.txt').decode()
        expected=next(line.split()[0] for line in sums.splitlines() if line.split()[-1]==filename)
        archive=fetch(url+filename)
        if hashlib.sha256(archive).hexdigest()!=expected:raise ValueError('Node checksum mismatch')
        with tarfile.open(fileobj=io.BytesIO(archive),mode='r:xz') as package:
            target.write_bytes(package.extractfile('node-v24.22.1-linux-x64/bin/node').read())
    target.chmod(0o755)
    (BASE/'public').mkdir(exist_ok=True)
    shutil.copyfile(BASE/'index.html',BASE/'public/index.html')
    if os.name=='nt':
        node=shutil.which('node')
        if not node:raise RuntimeError('Node build environment required')
        subprocess.run([node,str(BASE/'node_modules/esbuild/bin/esbuild'),'blob-browser.js',
            '--bundle','--format=esm','--platform=browser','--minify','--outfile=public/blob-client.js'],cwd=BASE,check=True)
    else:
        npm=shutil.which('npm')
        if not npm:raise RuntimeError('Node/npm build environment required')
        subprocess.run([npm,'run','bundle'],cwd=BASE,check=True)
    print(f'Frozen model verified: {len(files)} files. Browser upload module built.')

if __name__=='__main__':main()
