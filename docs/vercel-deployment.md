# Vercel deployment

Static files and model weights are built into `dist/` and served by the CDN. `/auth/*` and `/api/auth` use `api/handler.mjs`. No image inference runs on the server.

Environment: `APP_ORIGIN` (exact HTTPS production origin), `SESSION_SECRET` (32 cryptographically random bytes encoded as 64 hex characters), and the six existing provider client ID/secret variables. Never commit their values. Register the new `/auth/{google,kakao,naver}/callback` URLs at each provider.

On Vercel sessions are AES-256-GCM encrypted, authenticated, origin-bound HttpOnly/Secure/SameSite=Lax cookies, expiring after 8 hours. Provider access tokens are not persisted. OAuth transactions last 10 minutes; the browser replaces the transaction cookie on completion and the provider enforces authorization-code single use. Instances share only the environment key, not process memory. Logout clears this browser's cookie. A copied cookie cannot be centrally revoked before expiry; rotating SESSION_SECRET invalidates all cookies. Do not use this design for privileged server-side data without adding a durable revocation store.

Without SESSION_SECRET the local/Render memory-session behavior is retained. Vercel fails closed if the secret is absent.

`node --test tests/*.test.mjs` verifies auth, CSRF, expiration, tampering and cross-instance login. `node scripts/build-vercel.mjs` prepares public files. Check `/api/auth`, model downloads and all provider flows after deployment.
