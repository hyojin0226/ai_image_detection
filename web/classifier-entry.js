import './history.js?v=history-write';

const route = window.Fakit.routeName();
if (route === 'image' || route === 'video') {
  if (route === 'image') await import('./app.js?v=photo-stages');
  await import('./lens.js?v=history-write');
}
