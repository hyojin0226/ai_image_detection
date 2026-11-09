"use strict";
module.exports = function handler(request, response) {
  const value = process.env.VIDEO_API_BASE || "";
  let apiBase = "";
  try {
    const url = new URL(value);
    if (url.protocol === "https:" && !url.username && !url.password && url.pathname === "/" && !url.search && !url.hash) apiBase = url.origin;
  } catch { /* Missing configuration stays offline; no mock prediction. */ }
  let notebookApiBase = "";
  try {
    const url = new URL(process.env.NOTEBOOK_API_BASE || "");
    if (url.protocol === "https:" && /^[a-z0-9-]+\.trycloudflare\.com$/.test(url.hostname) && !url.port && !url.username && !url.password && url.pathname === "/" && !url.search && !url.hash) notebookApiBase = url.origin;
  } catch { /* Notebook service is optional; file analysis stays available. */ }
  response.setHeader("Cache-Control", "no-store");
  response.status(200).json({mode:"hybrid",inferenceLocation:"browser",configured:true,notebookApiBase,youtubeSource:apiBase,youtubeAutomaticAccessVerified:false});
};
