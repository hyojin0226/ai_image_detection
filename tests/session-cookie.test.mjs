import { test } from 'node:test';
import assert from 'node:assert/strict';
import { sessionCodec } from '../session-cookie.mjs';
import { createApp } from '../server.mjs';

const secret = 'ab'.repeat(32);
test('encrypted sessions survive instances, reject tampering, wrong origin and expiry', () => {
  const first = sessionCodec(secret, 'https://fakit.example', () => 100);
  const value = { user: { id: 'google:123', name: 'Test' }, csrf: 'csrf', expires: 200 };
  const token = first.seal(value);
  assert.ok(!token.includes('google'));
  assert.deepEqual(sessionCodec(secret, 'https://fakit.example', () => 100).open(token), value);
  const bytes = Buffer.from(token, 'base64url'); bytes[30] ^= 1;
  assert.equal(first.open(bytes.toString('base64url')), null);
  assert.equal(sessionCodec(secret, 'https://other.example', () => 100).open(token), null);
  assert.equal(sessionCodec(secret, 'https://fakit.example', () => 201).open(token), null);
  assert.equal(first.open('invalid'), null);
  assert.throws(() => sessionCodec('short', 'x'));
  assert.throws(() => createApp({ env: { VERCEL: '1' } }));
});

test('OAuth state and authenticated identity work across independent server instances', async t => {
  const env = { APP_ORIGIN: 'http://localhost:8000', SESSION_SECRET: secret, GOOGLE_CLIENT_ID: 'id', GOOGLE_CLIENT_SECRET: 'secret' };
  const fetchImpl = async url => Response.json(url.includes('/token') ? { access_token: 'private' } : { sub: '123', name: 'Tester' });
  async function instance() {
    const server = createApp({ env, fetchImpl });
    await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
    t.after(() => new Promise(resolve => server.close(resolve)));
    return (route, options = {}) => fetch(`http://127.0.0.1:${server.address().port}${route}`, { redirect: 'manual', ...options });
  }
  const a = await instance(), b = await instance();
  const start = await a('/auth/google/login');
  const pending = start.headers.get('set-cookie').split(';')[0];
  const state = new URL(start.headers.get('location')).searchParams.get('state');
  const wrong = await b('/auth/google/callback?code=x&state=wrong', { headers: { Cookie: pending } });
  assert.equal(wrong.headers.get('location'), '/?auth_error=invalid_state');
  const done = await b(`/auth/google/callback?code=x&state=${state}`, { headers: { Cookie: pending } });
  assert.equal(done.headers.get('location'), '/');
  const cookie = done.headers.get('set-cookie').split(';')[0];
  const auth = await (await a('/api/auth', { headers: { Cookie: cookie } })).json();
  assert.equal(auth.user.name, 'Tester');
  const codec = sessionCodec(secret, env.APP_ORIGIN);
  const sessionFor = id => 'session=' + codec.seal({ user: { id, name: 'Tester' }, csrf: 'new', expires: Date.now() + 60000 });
  const readAccess = async cookie => (await (await b('/api/auth', { headers: { Cookie: cookie } })).json()).history;
  assert.match(auth.history.scope, /^[a-f0-9]{64}$/);
  assert.notEqual(auth.history.key, auth.history.scope);
  assert.deepEqual(await readAccess(sessionFor('google:123')), auth.history);
  const other = await readAccess(sessionFor('google:456'));
  assert.notEqual(other.scope, auth.history.scope);
  assert.notEqual(other.key, auth.history.key);
  assert.equal((await (await b('/api/auth')).json()).history, null);
  assert.equal((await a('/auth/logout', { method: 'POST', headers: { Cookie: cookie } })).status, 403);
  const logout = await b('/auth/logout', { method: 'POST', headers: { Cookie: cookie, Origin: env.APP_ORIGIN, 'X-CSRF-Token': auth.csrf } });
  assert.equal(logout.status, 200);
  assert.match(logout.headers.get('set-cookie'), /Max-Age=0/);
});
