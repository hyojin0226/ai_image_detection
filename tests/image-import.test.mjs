import { test } from 'node:test';
import assert from 'node:assert/strict';
import { fetchImageFile } from '../web/image-import.js';

test('imports supported images without sending credentials or referrer', async () => {
  const file = await fetchImageFile('https://images.example/photo.png', { fetchImpl: async (url, options) => {
    assert.equal(url.hostname, 'images.example');
    assert.equal(options.credentials, 'omit');
    assert.equal(options.referrerPolicy, 'no-referrer');
    assert.equal(options.mode, 'cors');
    return new Response(new Uint8Array([1, 2, 3]), { headers: { 'Content-Type': 'image/png' } });
  } });
  assert.equal(file.type, 'image/png');
  assert.equal(file.size, 3);
});

test('rejects unsafe URL protocols and embedded credentials before fetching', async () => {
  for (const url of ['javascript:alert(1)', 'file:///test.png', 'data:image/png;base64,abc', 'https://user:pass@example.com/test.png']) {
    await assert.rejects(fetchImageFile(url, { fetchImpl: () => { assert.fail('must not fetch'); } }), /Invalid image URL/);
  }
});

test('rejects non-images, upstream errors, and bounded streaming overflow', async () => {
  await assert.rejects(fetchImageFile('https://example.com/a', { fetchImpl: async () => new Response('not an image', { headers: { 'Content-Type': 'text/html' } }) }), /Unsupported image/);
  await assert.rejects(fetchImageFile('https://example.com/a', { fetchImpl: async () => new Response('', { status: 403 }) }), /Image request failed/);
  await assert.rejects(fetchImageFile('https://example.com/a', { maxBytes: 2, fetchImpl: async () => new Response(new Uint8Array(3), { headers: { 'Content-Type': 'image/png' } }) }), /Image too large/);
  await assert.rejects(fetchImageFile('https://example.com/a', { maxBytes: 2, fetchImpl: async () => new Response('', { headers: { 'Content-Type': 'image/png', 'Content-Length': '3' } }) }), /Image too large/);
});
