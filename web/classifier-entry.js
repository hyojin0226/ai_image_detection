import './history.js?v=photo-result';

const route = window.Fakit.routeName();
if (route === 'image' || route === 'video') {
  if (route === 'image') await import('./app.js?v=convnext');
  await import('./lens.js?v=photo-result');
}
