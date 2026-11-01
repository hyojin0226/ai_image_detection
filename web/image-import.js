export async function fetchImageFile(rawUrl, { fetchImpl = fetch, signal, maxBytes = 15 * 1024 * 1024 } = {}) {
  const url = new URL(rawUrl);
  if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password || url.href.length > 16384) throw new Error('Invalid image URL');
  const controller = new AbortController();
  signal?.throwIfAborted();
  const abort = () => controller.abort();
  signal?.addEventListener('abort', abort, { once: true });
  const timeout = setTimeout(() => controller.abort(), 15000);
  try {
    const response = await fetchImpl(url, { mode: 'cors', credentials: 'omit', referrerPolicy: 'no-referrer', signal: controller.signal });
    if (!response.ok) throw new Error('Image request failed');
    const type = (response.headers.get('content-type') || '').split(';')[0].trim().toLowerCase();
    if (type.startsWith('video/')) { await response.body?.cancel(); throw Object.assign(new Error('영상 분석은 아직 지원하지 않아요.'), { code: 'VIDEO' }); }
    if (type === 'text/html') { await response.body?.cancel(); throw Object.assign(new Error('웹페이지 분석은 아직 지원하지 않아요.'), { code: 'PAGE' }); }
    if (!['image/jpeg', 'image/png', 'image/webp'].includes(type)) throw new Error('Unsupported image');
    if (Number(response.headers.get('content-length')) > maxBytes) throw new Error('Image too large');
    const reader = response.body.getReader();
    const chunks = [];
    let total = 0;
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      total += value.byteLength;
      if (total > maxBytes) { await reader.cancel(); throw new Error('Image too large'); }
      chunks.push(value);
    }
    return new File(chunks, `웹 이미지.${type.split('/')[1]}`, { type });
  } finally { clearTimeout(timeout); controller.abort(); signal?.removeEventListener('abort', abort); }
}
