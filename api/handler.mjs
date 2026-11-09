import { createApp } from '../server.mjs';
const app = createApp();
export default app.listeners('request')[0];
