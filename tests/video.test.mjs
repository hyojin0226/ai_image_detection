import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { pooled, head, featureGradient } from '../web/video-math.js';
import { analyze, classifyInput, LIMITS } from '../web/input-adapters.js';

test('video input calls only video model and enforces size and cancellation', async () => {
  const file = new File(['x'], 'test.mp4', { type: 'video/mp4' });
  const result = { status: 'result', real: .2, fake: .8, verdict: 'fake', video: {} };
  assert.equal(await analyze(classifyInput(file), { imageAnalyzer: () => assert.fail('image model used directly'), videoAnalyzer: async actual => { assert.equal(actual, file); return result; } }), result);
  await assert.rejects(analyze({ kind: 'video', file: { type:'video/mp4', size:LIMITS.videoBytes+1 } }, {videoAnalyzer:()=>assert.fail('oversized input')}), /90MB/);
  const controller=new AbortController();controller.abort();
  await assert.rejects(analyze(classifyInput(file), {signal:controller.signal,videoAnalyzer:()=>assert.fail('aborted')}), {name:'AbortError'});
});

test('frozen video heads and shared image model match published checksums', async () => {
  const config=JSON.parse(await readFile(new URL('../web/model/browser.json',import.meta.url)));
  for(const spec of config.members) {
    const data=await readFile(new URL('../web/model/'+spec.file,import.meta.url));
    assert.equal(createHash('sha256').update(data).digest('hex'),spec.sha256);
  }
  assert.equal(createHash('sha256').update(await readFile(new URL('../web/model/model.json',import.meta.url))).digest('hex'),config.source_image_sha256);
});

test('temporal pooling and analytical video gradient agree with finite differences', () => {
  const seq=Float64Array.from({length:8*257},(_,i)=>Math.sin(i*.13)*.2+i*.0001);
  const member={gamma:.001,mean:new Float64Array(772),scale:new Float64Array(772).fill(1),block:new Float64Array(772).fill(1),support:new Float64Array(772).fill(.05),coefficient:new Float64Array([.3])};
  const x=pooled(seq), predicted=head(x,[member],true), gradient=featureGradient(seq,x,predicted.gradient);
  assert.equal(x.length,772);assert.ok(predicted.score>0&&predicted.score<1);
  for(const index of [0,257,256,1027,2055]) {
    const a=seq.slice(),b=seq.slice(),epsilon=1e-5;a[index]+=epsilon;b[index]-=epsilon;
    const numerical=(head(pooled(a),[member]).score-head(pooled(b),[member]).score)/(2*epsilon);
    assert.ok(Math.abs(numerical-gradient[index])<1e-7,`${index}: ${numerical} vs ${gradient[index]}`);
  }
});
