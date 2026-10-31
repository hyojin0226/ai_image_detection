import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createApp } from '../server.mjs';

async function fixture(t, options = {}) {
  const server = createApp(options);
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  t.after(() => new Promise(resolve => server.close(resolve)));
  const base = `http://127.0.0.1:${server.address().port}`;
  return (route, options = {}) => fetch(base + route, { redirect: 'manual', ...options });
}
const env = { APP_ORIGIN: 'http://localhost:8000', GOOGLE_CLIENT_ID: 'test-id', GOOGLE_CLIENT_SECRET: 'test-secret', NAVER_CLIENT_ID: 'test-id', NAVER_CLIENT_SECRET: 'test-secret', KAKAO_CLIENT_ID: 'test-id' };
const cookieOf = response => response.headers.get('set-cookie').split(';')[0];

test('unconfigured providers, static assets and private paths', async t => {
  const request = await fixture(t, { env: {} });
  const data = await (await request('/api/auth')).json();
  assert.equal(data.user, null);
  assert.ok(data.providers.every(p => !p.enabled));
  assert.equal((await request('/auth/google/login')).headers.get('location'), '/?auth_error=not_configured');
  for (const route of ['/.env', '/server.mjs', '/README.md', '/model/../../.git/config']) assert.equal((await request(route)).status, 404);
  for (const route of ['/', '/ui.js', '/app.js', '/file-types.js', '/image-import.js', '/experience.css', '/icon.svg', '/extension.zip', '/model/inference_config.json']) assert.equal((await request(route)).status, 200);
});

for (const provider of ['google', 'naver', 'kakao']) {
  test(`${provider}: complete login, replay protection, CSRF and logout`, async t => {
    const calls = [];
    const profile = provider === 'google' ? { sub: '123', name: '<test>' } : provider === 'naver' ? { resultcode: '00', response: { id: '123', nickname: '<test>' } } : { id: 123, kakao_account: { profile: { nickname: '<test>' } } };
    const request = await fixture(t, { env, fetchImpl: async (url, options) => {
      calls.push({ url, options });
      return Response.json(calls.length === 1 ? { access_token: 'private-token' } : profile);
    } });
    const start = await request(`/auth/${provider}/login`);
    const pendingCookie = cookieOf(start);
    const target = new URL(start.headers.get('location'));
    assert.equal(target.searchParams.get('redirect_uri'), `http://localhost:8000/auth/${provider}/callback`);
    const callback = `/auth/${provider}/callback?code=test&state=${target.searchParams.get('state')}`;
    const wrong = await request(callback.replace(/state=.*/, 'state=wrong'), { headers: { Cookie: pendingCookie } });
    assert.equal(wrong.headers.get('location'), '/?auth_error=invalid_state');
    assert.equal(calls.length, 0);
    const success = await request(callback, { headers: { Cookie: pendingCookie } });
    assert.equal(success.headers.get('location'), '/');
    const cookie = cookieOf(success);
    assert.notEqual(cookie, pendingCookie);
    assert.match(success.headers.get('set-cookie'), /HttpOnly; SameSite=Lax/);
    const state = await (await request('/api/auth', { headers: { Cookie: cookie } })).json();
    assert.equal(state.user.id, `${provider}:123`);
    assert.equal(state.user.name, '<test>');
    assert.ok(!JSON.stringify(state).includes('private-token'));
    assert.equal(calls[0].options.body.get('code'), 'test');
    assert.equal(calls[1].options.headers.Authorization, 'Bearer private-token');
    assert.equal((await request(callback, { headers: { Cookie: pendingCookie } })).headers.get('location'), '/?auth_error=invalid_state');
    assert.equal((await request('/auth/logout', { method: 'POST', headers: { Cookie: cookie, Origin: env.APP_ORIGIN } })).status, 403);
    assert.equal((await request('/auth/logout', { method: 'POST', headers: { Cookie: cookie, Origin: 'https://evil.test', 'X-CSRF-Token': state.csrf } })).status, 403);
    assert.equal((await request('/auth/logout', { method: 'POST', headers: { Cookie: cookie, Origin: env.APP_ORIGIN, 'X-CSRF-Token': state.csrf } })).status, 200);
    assert.equal((await (await request('/api/auth', { headers: { Cookie: cookie } })).json()).user, null);
  });
}

test('expired transactions and provider failures never create a session', async t => {
  let time = 0;
  const request = await fixture(t, { env, now: () => time, fetchImpl: async () => { throw new Error('network failure'); } });
  const start = await request('/auth/google/login');
  const callback = `/auth/google/callback?code=test&state=${new URL(start.headers.get('location')).searchParams.get('state')}`;
  time = 600001;
  assert.equal((await request(callback, { headers: { Cookie: cookieOf(start) } })).headers.get('location'), '/?auth_error=invalid_state');
  const next = await request('/auth/google/login');
  const result = await request(`/auth/google/callback?code=test&state=${new URL(next.headers.get('location')).searchParams.get('state')}`, { headers: { Cookie: cookieOf(next) } });
  assert.equal(result.headers.get('location'), '/?auth_error=failed');
  assert.equal((await (await request('/api/auth', { headers: { Cookie: cookieOf(next) } })).json()).user, null);
});
