import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import vm from 'node:vm';

test('image context menu passes an encoded URL in the fragment only', async () => {
  let installed, clicked, menu, opened;
  const chrome = {
    runtime: { onInstalled: { addListener: fn => { installed = fn; } } },
    contextMenus: { removeAll: fn => fn(), create: value => { menu = value; }, onClicked: { addListener: fn => { clicked = fn; } } },
    tabs: { create: value => { opened = value; } },
  };
  vm.runInNewContext(await readFile(new URL('../extension/background.js', import.meta.url), 'utf8'), { chrome, URL, URLSearchParams });
  installed();
  assert.equal(menu.contexts[0], 'image');
  const source = 'https://images.example/photo.jpg?a=1&b=한글#detail';
  clicked({ menuItemId: menu.id, srcUrl: source });
  const target = new URL(opened.url);
  assert.equal(target.origin, 'https://fakit-lens.vercel.app');
  assert.equal(target.search, '');
  assert.equal(new URLSearchParams(target.hash.slice(1)).get('image'), new URL(source).href);
  for (const srcUrl of ['blob:https://example.com/id', 'data:image/png;base64,a', 'javascript:alert(1)', 'https://user:pass@example.com/a.jpg']) {
    clicked({ menuItemId: menu.id, srcUrl });
    assert.equal(new URL(opened.url).hash, '#import_error=unsupported');
  }
  opened = null;
  clicked({ menuItemId: 'other', srcUrl: source });
  assert.equal(opened, null);
  const manifest = JSON.parse(await readFile(new URL('../extension/manifest.json', import.meta.url), 'utf8'));
  assert.equal(manifest.manifest_version, 3);
  assert.deepEqual(manifest.permissions, ['contextMenus']);
});
