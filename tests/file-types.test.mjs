import { test } from 'node:test';
import assert from 'node:assert/strict';
import { classifyFile, supportedImageMime } from '../web/file-types.js';

test('one input routes image, video, and text files independently of supported models', () => {
  assert.equal(classifyFile({ name: 'photo.jpg', type: 'image/jpeg' }), 'image');
  assert.equal(classifyFile({ name: 'clip.mp4', type: 'video/mp4' }), 'video');
  assert.equal(classifyFile({ name: 'essay.txt', type: 'text/plain' }), 'text');
  assert.equal(classifyFile({ name: 'essay.pdf', type: 'application/pdf' }), 'text');
  assert.equal(classifyFile({ name: 'clip.MOV', type: '' }), 'video');
  assert.equal(classifyFile({ name: 'archive.zip', type: 'application/zip' }), 'unknown');
  assert.equal(classifyFile({ name: 'pretends.jpg', type: 'video/mp4' }), 'video');
});
test('recognizing an image does not automatically enable an unsupported decoder', () => {
  assert.equal(classifyFile({ name: 'photo.heic', type: '' }), 'image');
  assert.equal(supportedImageMime({ name: 'photo.heic', type: '' }), null);
  assert.equal(supportedImageMime({ name: 'photo.PNG', type: '' }), 'image/png');
  assert.equal(supportedImageMime({ name: 'pretends.png', type: 'application/zip' }), null);
});
