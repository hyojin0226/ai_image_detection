import { mkdir, readdir, copyFile } from 'node:fs/promises';
// Copy only the public application, never notebooks, secrets or repository docs.
await mkdir('dist/model', { recursive: true });
for (const name of await readdir('web')) {
  if (/\.(html|js|css|svg|ico|zip)$/.test(name)) await copyFile(`web/${name}`, `dist/${name}`);
}
for (const name of await readdir('web/model')) {
  if (/\.(json|bin)$/.test(name)) await copyFile(`web/model/${name}`, `dist/model/${name}`);
}
console.log('Public assets ready in dist/');
