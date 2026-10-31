export async function fetchImageFile(rawUrl, { fetchImpl = fetch, maxBytes = 15 * 1024 * 1024 } = {}) {
  const url = new URL(rawUrl);
  if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password || url.href.length > 16384) throw new Error('Invalid image URL');
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 15000);
  try {
    const response = await fetchImpl(url, { mode: 'cors', credentials: 'omit', referrerPolicy: 'no-referrer', signal: controller.signal });
    if (!response.ok) throw new Error('Image request failed');
    const type = (response.headers.get('content-type') || '').split(';')[0].trim().toLowerCase();
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
  } finally { clearTimeout(timeout); controller.abort(); }
}

if (typeof document !== 'undefined') {
  const input = document.getElementById('image-url');
  const status = document.getElementById('import-status');
  // Consume the extension payload before the hash router navigates anywhere else.
  const fragment = new URLSearchParams(location.hash.slice(1));
  if (fragment.has('image') || fragment.has('import_error')) {
    input.value = (fragment.get('image') || '').slice(0, 16384);
    document.getElementById('url-import').open = true;
    status.textContent = fragment.has('import_error')
      ? '이 주소는 가져올 수 없어요. 이미지를 저장한 뒤 직접 업로드해 주세요.'
      : '확장 프로그램에서 주소를 가져왔어요. 가져와 분석하기를 눌러주세요.';
    history.replaceState(null, '', location.pathname + location.search + '#/');
    document.getElementById('url-import').scrollIntoView({ block: 'center' });
  }
  document.getElementById('image-url-form').addEventListener('submit', async event => {
    event.preventDefault();
    const button = document.getElementById('import-image');
    button.disabled = true;
    status.textContent = '이미지를 가져오는 중…';
    try {
      const file = await fetchImageFile(input.value);
      await window.Fakit.addFiles([file]);
      status.textContent = '이미지를 분석 목록에 추가했어요.';
    } catch {
      status.textContent = '이미지를 가져오지 못했어요. 원본 사이트의 접근 제한 또는 지원하지 않는 형식·크기일 수 있습니다. 15MB 이하 JPG·PNG·WEBP 사진을 저장한 뒤 직접 업로드해 주세요.';
    } finally { button.disabled = false; }
  });
}
