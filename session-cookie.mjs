import { createCipheriv, createDecipheriv, randomBytes } from 'node:crypto';

// Authenticated encryption: neither identity nor CSRF values are readable by clients.
export function sessionCodec(secret, audience, now = Date.now) {
  if (!/^[a-f0-9]{64}$/i.test(secret || '')) throw new Error('SESSION_SECRET must be 32 random bytes encoded as hex.');
  const key = Buffer.from(secret, 'hex');
  const aad = Buffer.from(`fakit-session-v1:${audience}`);
  return {
    seal(value) {
      const iv = randomBytes(12);
      const cipher = createCipheriv('aes-256-gcm', key, iv);
      cipher.setAAD(aad);
      const body = Buffer.concat([cipher.update(JSON.stringify(value), 'utf8'), cipher.final()]);
      return Buffer.concat([iv, cipher.getAuthTag(), body]).toString('base64url');
    },
    open(token) {
      try {
        if (typeof token !== 'string' || token.length > 3800) return null;
        const bytes = Buffer.from(token, 'base64url');
        if (bytes.length < 29) return null;
        const decipher = createDecipheriv('aes-256-gcm', key, bytes.subarray(0, 12));
        decipher.setAAD(aad);
        decipher.setAuthTag(bytes.subarray(12, 28));
        const value = JSON.parse(Buffer.concat([decipher.update(bytes.subarray(28)), decipher.final()]).toString('utf8'));
        return Number.isFinite(value.expires) && value.expires > now() ? value : null;
      } catch { return null; }
    },
  };
}
