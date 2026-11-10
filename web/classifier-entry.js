import './history.js?v=result-summary';

const route = window.Fakit.routeName();
if (route === 'image' || route === 'video') {
  if (route === 'image') await import('./app.js?v=classifiers');
  await import('./lens.js?v=video-preview');
}
