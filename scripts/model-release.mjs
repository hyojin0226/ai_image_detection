import {readdir,readFile,writeFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
export async function modelRelease() {
  const hash=createHash('sha256');
  for(const name of (await readdir('web/model')).filter(n=>/\.(json|bin)$/.test(n)).sort()) {
    const bytes=await readFile('web/model/'+name);
    hash.update(name);hash.update(name.endsWith('.json')?JSON.stringify(JSON.parse(bytes)):bytes);
  }
  return 'r-'+hash.digest('hex').slice(0,20);
}
const release=await modelRelease();
await writeFile('web/model-release.js',`// Generated from all model configs and weights. Never edit paths independently.\nexport const MODEL_BASE = '/model/${release}/';\n`);
console.log(release);
