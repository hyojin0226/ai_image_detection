import { test } from 'node:test';
import assert from 'node:assert/strict';
import { analyzeBatch } from '../web/batch-input.js';

test('batch preserves order and continues after an unsupported input or decode failure', async () => {
  const events=[];let active=0;
  const results=await analyzeBatch([1,2,3,4],{onItem:(index,state)=>events.push([index,state]),analyzeImpl:async input=>{
    assert.equal(active++,0);await Promise.resolve();active--;
    if(input===2)throw new Error('읽기 실패');
    return input===3?{status:'unsupported',message:'영상 미지원'}:{status:'result',real:input/10};
  }});
  assert.deepEqual(results.map(r=>r.status),['result','error','unsupported','result']);
  assert.deepEqual(events.filter(([,s])=>s==='analyzing').map(([i])=>i),[0,1,2,3]);
  assert.equal(results[3].real,.4);
});
test('abort stops a batch before the next image and suppresses late completion', async () => {
  const controller=new AbortController();let calls=0;const completed=[];
  await assert.rejects(analyzeBatch([1,2,3],{signal:controller.signal,onItem:(i,s)=>{if(s==='result')completed.push(i);},analyzeImpl:async()=>{
    calls++;controller.abort();return {status:'result',real:.5};
  }}),{name:'AbortError'});
  assert.equal(calls,1);assert.deepEqual(completed,[]);
});
