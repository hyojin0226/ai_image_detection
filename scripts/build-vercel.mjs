import { mkdir, readdir, copyFile, rm } from 'node:fs/promises';
import path from 'node:path';
import './model-release.mjs';
const {MODEL_BASE}=await import('../web/model-release.js?build='+Date.now());
// Copy only the public application, never notebooks, secrets or repository docs.
const output=path.resolve('dist');
if(path.dirname(output)!==process.cwd()||path.basename(output)!=='dist')throw Error('Unexpected build path');
await rm(output,{recursive:true,force:true});
await mkdir('dist'+MODEL_BASE, { recursive: true });
for (const name of await readdir('web')) {
  if (/\.(html|js|css|svg|ico|zip)$/.test(name)) await copyFile(`web/${name}`, `dist/${name}`);
}
for (const name of await readdir('web/model')) {
  if (/\.(json|bin|onnx|tflite)$/.test(name)) await copyFile(`web/model/${name}`, `dist${MODEL_BASE}${name}`);
}
console.log('Public assets ready in dist/');
