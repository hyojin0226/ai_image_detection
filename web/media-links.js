export function youtubeVideoId(raw) {
  let url; try { url = new URL(raw); } catch { return null; }
  if (!['https:', 'http:'].includes(url.protocol) || url.username || url.password) return null;
  const host = url.hostname.toLowerCase();
  let id;
  if (host === 'youtu.be') id = url.pathname.split('/')[1];
  else if (['youtube.com', 'www.youtube.com', 'm.youtube.com'].includes(host)) {
    id = url.pathname === '/watch' ? url.searchParams.get('v') : /^\/(?:shorts|embed)\/([^/]+)\/?$/.exec(url.pathname)?.[1];
  }
  return /^[\w-]{11}$/.test(id || '') ? id : null;
}
