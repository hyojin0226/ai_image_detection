import './history.js?v=original-photo';

const route = window.Fakit.routeName();
if (route === 'image' || route === 'video') {
  if (route === 'image') await import('./app.js?v=convnext');
  await import('./lens.js?v=original-photo');
}
