// Change this to your deployed Fakit URL when publishing the site.
const FAKIT_URL = 'http://localhost:8000/';
const MENU_ID = 'fakit-image';

chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.removeAll(() => {
    chrome.contextMenus.create({
      id: MENU_ID,
      title: 'Fakit으로 이미지 가져오기',
      contexts: ['image'],
    });
  });
});

chrome.contextMenus.onClicked.addListener((info) => {
  if (info.menuItemId !== MENU_ID) return;
  const target = new URL(FAKIT_URL);
  try {
    const source = new URL(info.srcUrl);
    if (!['https:', 'http:'].includes(source.protocol) || source.username || source.password || source.href.length > 16384) throw new Error('Unsupported image URL');
    // Fragments do not reach the Fakit HTTP server or its access logs.
    target.hash = new URLSearchParams({ image: source.href }).toString();
  } catch {
    target.hash = 'import_error=unsupported';
  }
  chrome.tabs.create({ url: target.href });
});
