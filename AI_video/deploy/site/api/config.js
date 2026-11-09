"use strict";
module.exports = function handler(request, response) {
  const value = process.env.VIDEO_API_BASE || "";
  let apiBase = "";
  try {
    const url = new URL(value);
    if (url.protocol === "https:" && !url.username && !url.password && url.pathname === "/" && !url.search && !url.hash) apiBase = url.origin;
  } catch { /* Missing configuration stays offline; no mock prediction. */ }
  response.setHeader("Cache-Control", "no-store");
  response.status(200).json({mode:"browser",inferenceLocation:"browser",configured:true,youtubeSource:apiBase,youtubeAutomaticAccessVerified:false});
};
