import {test} from 'node:test';
import assert from 'node:assert/strict';
import {once} from 'node:events';
import {readFile,readdir} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {createApp} from '../server.mjs';
import {MODEL_BASE} from '../web/model-release.js';

test('one content-addressed release contains matching manifest, config and shards',async()=>{
  const hash=createHash('sha256');
  for(const name of (await readdir('web/model')).filter(n=>/\.(json|bin)$/.test(n)).sort()){
    const bytes=await readFile('web/model/'+name);
    hash.update(name);hash.update(name.endsWith('.json')?JSON.stringify(JSON.parse(bytes)):bytes);
  }
  assert.equal(MODEL_BASE,`/model/r-${hash.digest('hex').slice(0,20)}/`);
  const model=JSON.parse(await readFile('web/model/model.json'));
  for(const group of model.weightsManifest)for(const file of group.paths){
    assert.match(file,/^[\w.-]+\.bin$/); assert.ok((await readFile('web/model/'+file)).length>0);
  }
});
test('versioned public model routes cache without exposing arbitrary repository files',async t=>{
  const server=createApp({env:{}});server.listen(0,'127.0.0.1');await once(server,'listening');
  t.after(()=>server.close());const base=`http://127.0.0.1:${server.address().port}`;
  const config=await fetch(base+MODEL_BASE+'inference_config.json');assert.equal(config.status,200);
  assert.match(config.headers.get('cache-control'),/max-age=604800/);
  assert.equal((await config.json()).run_name,'efficientnetv2s_experiment_01');
  assert.equal((await fetch(base+'/model/r-00000000000000000000/model.json')).status,404);
  assert.equal((await fetch(base+'/.env')).status,404);
  const js=await fetch(base+'/inference-worker.js');assert.equal(js.status,200);
  assert.match(js.headers.get('cache-control'),/max-age=300, stale-while-revalidate/);
});
