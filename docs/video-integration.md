# Video model integration

Source: `hyojin0226/ai_image_detection`, branch `study/KongDuHyun`, commit `9730b30` (browser migration).

`web/browser-engine.js` and `web/video-math.js` come from `AI_video/deploy/site`. The team published `browser.json`, `head-1.bin`, `head-2.bin`, `suffix.json`, `suffix.bin` at `https://duhyeon-ai-video.vercel.app/model/`. Head checksums and the original image model checksum are verified in tests. No retraining or threshold changes. The 1st image model is shared, not replaced by a 2nd experiment model.

Adaptations: reuse Fakit's already loaded image backbone, avoid switching TensorFlow backends under an active image model, and allow scheduled work to advance when requestAnimationFrame is throttled. UI uses the unified input, exposes real extraction/Grad-CAM progress, eight segment scores, eight major frames and remaining frames, cancel/reset, and fullscreen frame comparisons. File input remains local (3–40 seconds, at most 90 MiB, browser-decodable MP4/WebM). Video results are not persisted in image history.

The classifier aggregates 257 features per frame into 772 temporal statistics; two fixed RBF heads are averaged. AI score >= 0.5 means fake. This is a model score, not calibrated probability. Browser decoding/resampling differs from FFmpeg and has no independent accuracy validation. Grad-CAM supports the selected class; it does not locate proven manipulation.

YouTube automatic retrieval is NOT integrated. The team's `/api/source` accepts only their frontend Origin and reports automatic retrieval as unverified. Fakit does not spoof that Origin or send user credentials. YouTube links receive an explicit external-tool option or instructions to select a local file. Direct local video inference does not need the team's notebook or external analysis server.
