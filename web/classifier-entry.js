import './history.js';

const route = window.Fakit.routeName();
if (route === 'image' || route === 'video') {
  if (route === 'image') await import('./app.js?v=classifiers');
  await import('./lens.js?v=classifiers');
}
