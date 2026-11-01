import http from 'node:http';
import { randomBytes, timingSafeEqual } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const webRoot = fileURLToPath(new URL('./web/', import.meta.url));
const providers = {
  kakao: { authorize: 'https://kauth.kakao.com/oauth/authorize', token: 'https://kauth.kakao.com/oauth/token', profile: 'https://kapi.kakao.com/v2/user/me' },
  naver: { authorize: 'https://nid.naver.com/oauth2.0/authorize', token: 'https://nid.naver.com/oauth2.0/token', profile: 'https://openapi.naver.com/v1/nid/me' },
  google: { authorize: 'https://accounts.google.com/o/oauth2/v2/auth', token: 'https://oauth2.googleapis.com/token', profile: 'https://openidconnect.googleapis.com/v1/userinfo', scope: 'openid profile' },
};
const random = () => randomBytes(32).toString('hex');
const equal = (a, b) => typeof a === 'string' && typeof b === 'string' && Buffer.byteLength(a) === Buffer.byteLength(b) && timingSafeEqual(Buffer.from(a), Buffer.from(b));
const sessionTTL = 8 * 60 * 60 * 1000;

export function createApp({ env = process.env, fetchImpl = fetch, now = Date.now } = {}) {
  const origin = new URL(env.APP_ORIGIN || env.RENDER_EXTERNAL_URL || 'http://localhost:8000');
  if (origin.pathname !== '/' || origin.search || origin.hash || origin.username || origin.password ||
      (origin.protocol !== 'https:' && !(origin.protocol === 'http:' && ['localhost', '127.0.0.1', '[::1]'].includes(origin.hostname)))) {
    throw new Error('APP_ORIGIN must be an HTTPS origin (HTTP is allowed only on loopback).');
  }
  const secure = origin.protocol === 'https:';
  const cookieName = secure ? '__Host-session' : 'session';
  const sessions = new Map();
  const credentials = (name) => ({ id: env[`${name.toUpperCase()}_CLIENT_ID`], secret: env[`${name.toUpperCase()}_CLIENT_SECRET`] });
  const enabled = (name) => { const c = credentials(name); return Boolean(c.id && (name === 'kakao' || c.secret)); };
  const cookie = (id, age) => `${cookieName}=${id}; Path=/; HttpOnly; SameSite=Lax; Max-Age=${age}${secure ? '; Secure' : ''}`;
  async function remote(url, options) {
    const response = await fetchImpl(url, { ...options, redirect: 'error', signal: AbortSignal.timeout(10000) });
    if (!response.ok) throw new Error('Provider request failed');
    const data = await response.json();
    if (!data || data.error) throw new Error('Invalid provider response');
    return data;
  }
  return http.createServer(async (req, res) => {
    res.setHeader('X-Content-Type-Options', 'nosniff');
    res.setHeader('Referrer-Policy', 'no-referrer');
    const send = (status, data) => { res.writeHead(status, { 'Content-Type': 'application/json; charset=utf-8' }); res.end(JSON.stringify(data)); };
    const redirect = (location) => { res.writeHead(302, { Location: location }); res.end(); };
    try {
      const url = new URL(req.url, origin);
      if (url.pathname.startsWith('/auth/') || url.pathname.startsWith('/api/')) {
        res.setHeader('Cache-Control', 'no-store');
        for (const [key, value] of sessions) if (value.expires <= now()) sessions.delete(key);
        const sid = req.headers.cookie?.split(';').map(s => s.trim()).find(s => s.startsWith(`${cookieName}=`))?.slice(cookieName.length + 1);
        const session = sessions.get(sid);
        if (req.method === 'GET' && url.pathname === '/api/auth') {
          return send(200, { user: session?.user || null, csrf: session?.csrf || null, providers: Object.keys(providers).map(id => ({ id, enabled: enabled(id) })) });
        }
        if (req.method === 'POST' && url.pathname === '/auth/logout') {
          if (req.headers.origin !== origin.origin || !session || !equal(req.headers['x-csrf-token'], session.csrf)) return send(403, { error: 'Forbidden' });
          sessions.delete(sid);
          res.setHeader('Set-Cookie', cookie('', 0));
          return send(200, { ok: true });
        }
        const match = url.pathname.match(/^\/auth\/(kakao|naver|google)\/(login|callback)$/);
        if (req.method !== 'GET' || !match) return send(404, { error: 'Not found' });
        const [, name, action] = match;
        if (!enabled(name)) return redirect('/?auth_error=not_configured');
        const p = providers[name];
        const c = credentials(name);
        const callback = `${origin.origin}/auth/${name}/callback`;
        if (action === 'login') {
          if (sessions.size >= 10000) return send(503, { error: 'Please try again later' });
          const id = random();
          const state = random();
          // A fresh browser-bound transaction also prevents session fixation.
          sessions.delete(sid);
          sessions.set(id, { expires: now() + 10 * 60 * 1000, pending: { name, state } });
          res.setHeader('Set-Cookie', cookie(id, 600));
          const target = new URL(p.authorize);
          target.search = new URLSearchParams({ response_type: 'code', client_id: c.id, redirect_uri: callback, state, ...(p.scope ? { scope: p.scope } : {}) }).toString();
          return redirect(target.href);
        }
        const pending = session?.pending;
        if (!pending || pending.name !== name || !equal(url.searchParams.get('state'), pending.state)) return redirect('/?auth_error=invalid_state');
        sessions.delete(sid); // Consume once, even on denial or upstream failure.
        res.setHeader('Set-Cookie', cookie('', 0));
        if (url.searchParams.has('error')) return redirect('/?auth_error=cancelled');
        const code = url.searchParams.get('code');
        if (!code) return redirect('/?auth_error=failed');
        try {
          const token = await remote(p.token, { method: 'POST', body: new URLSearchParams({ grant_type: 'authorization_code', client_id: c.id, ...(c.secret ? { client_secret: c.secret } : {}), code, redirect_uri: callback, state: pending.state }) });
          if (typeof token.access_token !== 'string' || !token.access_token) throw new Error('Missing token');
          const profile = await remote(p.profile, { headers: { Authorization: `Bearer ${token.access_token}` } });
          const value = name === 'naver' ? profile.response : profile;
          const id = name === 'google' ? value?.sub : value?.id;
          if (!id || (name === 'naver' && profile.resultcode !== '00')) throw new Error('Missing identity');
          const displayName = name === 'kakao' ? value.kakao_account?.profile?.nickname || value.properties?.nickname : value.name || value.nickname;
          const newSid = random();
          sessions.set(newSid, { expires: now() + sessionTTL, csrf: random(), user: { id: `${name}:${id}`, provider: name, name: typeof displayName === 'string' ? displayName.slice(0, 100) : '사용자' } });
          res.setHeader('Set-Cookie', cookie(newSid, sessionTTL / 1000));
          return redirect('/');
        } catch {
          // Do not log provider responses: they can contain credentials or personal data.
          return redirect('/?auth_error=failed');
        }
      }
      if (!['GET', 'HEAD'].includes(req.method)) return send(405, { error: 'Method not allowed' });
      const pathname = decodeURIComponent(url.pathname);
      // Explicit public-file allowlist: repository files and credentials are never served.
      if (!/^\/(?:|index\.html|(?:app|ui|lens|file-types|image-import|input-adapters|batch-input)\.js|(?:style|experience)\.css|icon\.(?:svg|ico)|extension\.zip|model\/[\w.-]+\.(?:json|bin))$/.test(pathname)) return send(404, { error: 'Not found' });
      const file = path.join(webRoot, pathname === '/' ? 'index.html' : pathname.slice(1));
      const content = await readFile(file);
      const types = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.css': 'text/css; charset=utf-8', '.json': 'application/json', '.bin': 'application/octet-stream', '.svg': 'image/svg+xml', '.ico': 'image/x-icon', '.zip': 'application/zip' };
      res.writeHead(200, { 'Content-Type': types[path.extname(file)] });
      res.end(req.method === 'HEAD' ? undefined : content);
    } catch (error) {
      send(error.code === 'ENOENT' ? 404 : 400, { error: 'Request could not be completed' });
    }
  });
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const server = createApp();
  server.listen(Number(process.env.PORT || 8000), process.env.HOST || '127.0.0.1', () => console.log(`AI Face Detector: ${process.env.APP_ORIGIN || 'http://localhost:8000'}`));
}
