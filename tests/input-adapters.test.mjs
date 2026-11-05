import { test } from 'node:test';
import assert from 'node:assert/strict';
import { analyze, classifyInput, LIMITS } from '../web/input-adapters.js';
import { fetchImageFile } from '../web/image-import.js';

test('MIME wins over filename; short standalone links are distinguished from prose', () => {
  assert.equal(classifyInput(new File(['x'], 'photo.png', { type: 'video/mp4' })).kind, 'video');
  assert.equal(classifyInput('https://example.org/a').kind, 'url');
  assert.equal(classifyInput('이 주소를 봐요 https://example.org/a').kind, 'text');
  assert.equal(classifyInput('https://example.org/' + 'a'.repeat(200)).kind, 'url');
});
test('absent models never call image inference or fabricate scores', async () => {
  for (const kind of ['text','video','unknown']) {
    const result = await analyze({ kind, value:'테스트' }, { imageAnalyzer: () => assert.fail('wrong model') });
    assert.equal(result.status, 'unsupported');
    assert.equal(result.real, undefined);
  }
});
test('image payload is passed unchanged; existing score and verdict are retained', async () => {
  const file = new File(['x'], 'test.png', {type:'image/png'});
  const score = { status:'result', real:.3, fake:.7, verdict:'real' };
  const result = await analyze(classifyInput(file,'picker'), { imageAnalyzer: async actual => { assert.equal(actual,file); return score; } });
  assert.equal(result,score);
});
test('limits and abort stop inference', async () => {
  assert.equal((await analyze({kind:'text',value:'a'.repeat(20001)})).status, 'unsupported');
  await assert.rejects(analyze({kind:'image',file:{type:'image/png',size:LIMITS.imageBytes+1}}), /15MB/);
  const controller = new AbortController(); controller.abort();
  await assert.rejects(analyze({kind:'text',value:'a'},{signal:controller.signal}), {name:'AbortError'});
});
test('video/page URL reports unsupported without downloading its body', async () => {
  for (const type of ['video/mp4','text/html']) {
    let cancelled=false;
    const result=await analyze({kind:'url',value:'https://example.org/file'}, {fetchImpl:async()=>new Response(new ReadableStream({cancel(){cancelled=true;}}),{headers:{'content-type':type}})});
    assert.equal(result.status,'unsupported'); assert.equal(cancelled,true);
  }
});
test('cancelling URL import forwards the abort signal', async () => {
  const controller=new AbortController();
  const pending=fetchImageFile('https://example.org/image',{signal:controller.signal,fetchImpl:(_url,{signal})=>new Promise((_resolve,reject)=>signal.addEventListener('abort',()=>reject(new DOMException('Aborted','AbortError'))))});
  controller.abort(); await assert.rejects(pending,{name:'AbortError'});
});
