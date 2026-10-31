// File routing is independent of model availability. No file contents are sent.
export function classifyFile(file) {
  const mime = (file.type || '').toLowerCase();
  const ext = (file.name || '').split('.').pop().toLowerCase();
  if (mime.startsWith('image/')) return 'image';
  if (mime.startsWith('video/')) return 'video';
  if (mime.startsWith('text/') || ['application/pdf', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'].includes(mime)) return 'text';
  if (mime && mime !== 'application/octet-stream') return 'unknown';
  if (['jpg', 'jpeg', 'png', 'webp', 'gif', 'bmp', 'avif', 'heic'].includes(ext)) return 'image';
  if (['mp4', 'webm', 'mov', 'm4v', 'avi'].includes(ext)) return 'video';
  if (['txt', 'md', 'pdf', 'docx'].includes(ext)) return 'text';
  return 'unknown';
}

export function supportedImageMime(file) {
  const mime = (file.type || '').toLowerCase();
  if (['image/jpeg', 'image/png', 'image/webp'].includes(mime)) return mime;
  if (mime && mime !== 'application/octet-stream') return null;
  return { jpg: 'image/jpeg', jpeg: 'image/jpeg', png: 'image/png', webp: 'image/webp' }[(file.name || '').split('.').pop().toLowerCase()] || null;
}
