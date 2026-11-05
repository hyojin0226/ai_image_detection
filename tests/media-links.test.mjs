import { test } from 'node:test';
import assert from 'node:assert/strict';
import { youtubeVideoId } from '../web/media-links.js';
import { analyze } from '../web/input-adapters.js';
test('YouTube watch, shorts and share URLs are recognized without trusting lookalike hosts', () => {
  for(const url of ['https://youtu.be/abcdefghijk?t=3','https://www.youtube.com/shorts/abcdefghijk','https://m.youtube.com/watch?v=abcdefghijk']) assert.equal(youtubeVideoId(url),'abcdefghijk');
  for(const url of ['https://youtube.com.evil.test/watch?v=abcdefghijk','https://youtube.com@evil.test/shorts/abcdefghijk','https://youtube.com/shorts/abc']) assert.equal(youtubeVideoId(url),null);
});
test('YouTube input never masquerades a thumbnail as video inference', async () => {
  const result=await analyze({kind:'url',value:'https://youtube.com/shorts/abcdefghijk'}, {fetchImpl:()=>assert.fail('no video retrieval service'),imageAnalyzer:()=>assert.fail('not a video model')});
  assert.equal(result.status,'unsupported'); assert.match(result.message,/유튜브/); assert.equal(result.real,undefined);
});
