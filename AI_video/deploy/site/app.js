"use strict";
const $ = (id) => document.getElementById(id);
let mode = "file", apiBase = "", online = false, busy = false, detailsRendered = false, result = null;
let selectedFile = null, fileReady = false, previewUrl = "", selection = 0, connecting = false;
const availability = $("availability").textContent;
function refreshButton() {
  $("analyze").disabled = !online || busy || (mode === "file" ? !fileReady : !$("video-url").value.trim());
}
const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));
function error(message) { $("error").textContent = message; $("error").hidden = !message; }
function status(message) {
  $("progress-message").textContent = message;
  const count = message.match(/Grad-CAM\s+(\d+)\/64/);
  if (count) { $("frame-progress").hidden = false; $("frame-progress").value = Math.min(64, Number(count[1])); }
}
async function api(path, options = {}) {
  const response = await fetch(apiBase + path, {cache:"no-store", ...options});
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "분석 서버에 연결하지 못했습니다.");
  return data;
}
async function connect() {
  if (connecting) return;
  connecting = true; $("reconnect").disabled = true;
  try {
    const response = await fetch("/api/config", {cache:"no-store", signal:AbortSignal.timeout(10000)});
    if (!response.ok) throw new Error("연결 설정 확인 실패");
    const config = await response.json();
    if (!config.configured) throw new Error("분석 서버 연결 준비 중");
    apiBase = config.apiBase || "";
    await api("/health", {signal:AbortSignal.timeout(12000)});
    online = true;
    $("connection").textContent = "● 분석 서버 연결됨";
    $("connection").className = "status online";
    $("availability").textContent = availability;
  } catch (cause) {
    online = false;
    $("connection").textContent = "● 분석 서버 오프라인";
    $("connection").className = "status offline";
    $("availability").textContent = "분석 서버가 연결되지 않았습니다. 운영자가 노트북 분석 서버를 실행하면 이용할 수 있습니다.";
  }
  connecting = false; $("reconnect").disabled = false; $("reconnect").hidden = online;
  refreshButton();
}
function tab(next) {
  mode = next;
  for (const name of ["file", "url"]) {
    $(name + "-tab").setAttribute("aria-selected", String(name === mode));
    $(name + "-panel").hidden = name !== mode;
  }
  error("");
  refreshButton();
}
$("file-tab").addEventListener("click", () => tab("file"));
$("url-tab").addEventListener("click", () => tab("url"));
function chooseFile(file) {
  if (busy) return;
  const current = ++selection;
  error(""); selectedFile = file || null; fileReady = false;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  const video = $("video-preview");
  video.onloadedmetadata = null; video.onerror = null; video.removeAttribute("src"); video.load();
  $("preview").hidden = !file;
  $("filename").textContent = file ? `${file.name} · ${(file.size / 1024 ** 2).toFixed(1)} MB` : "선택한 파일이 없습니다.";
  refreshButton();
  if (!file) return;
  if (!file.size || file.size > 90 * 1024 ** 2) { error("90 MB 이하의 영상 파일을 선택하세요."); $("preview").hidden = true; return; }
  $("preview-name").textContent = file.name; $("preview-meta").textContent = "영상 길이를 확인하고 있습니다…";
  video.onloadedmetadata = () => {
    if (selection !== current) return;
    const duration = video.duration;
    if (!Number.isFinite(duration) || duration < 3 || duration > 40) {
      error("3~40초 영상을 선택하세요. 선택한 영상: " + (Number.isFinite(duration) ? duration.toFixed(1) + "초" : "길이 확인 불가"));
      $("preview-meta").textContent = "지원하는 길이의 영상을 다시 선택해 주세요."; fileReady = false;
    } else {
      $("preview-meta").textContent = `${duration.toFixed(1)}초 · ${video.videoWidth} × ${video.videoHeight} · ${(file.size / 1024 ** 2).toFixed(1)} MB`;
      fileReady = true;
    }
    refreshButton();
  };
  video.onerror = () => {
    if (selection !== current) return;
    $("preview-meta").textContent = "브라우저 미리보기를 지원하지 않는 형식입니다. 분석 서버에서 형식과 길이를 확인합니다.";
    fileReady = true; refreshButton();
  };
  previewUrl = URL.createObjectURL(file); video.src = previewUrl;
}
$("video-file").addEventListener("change", () => chooseFile($("video-file").files[0]));
$("remove-file").addEventListener("click", () => { $("video-file").value = ""; chooseFile(null); });
$("video-url").addEventListener("input", refreshButton);
$("reconnect").addEventListener("click", connect);
for (const name of ["dragenter", "dragover"]) $("drop-zone").addEventListener(name, event => { event.preventDefault(); if (!busy) $("drop-zone").classList.add("dragging"); });
for (const name of ["dragleave", "drop"]) $("drop-zone").addEventListener(name, event => { event.preventDefault(); $("drop-zone").classList.remove("dragging"); });
$("drop-zone").addEventListener("drop", event => {
  if (busy) return;
  if (event.dataTransfer.files.length !== 1) { error("영상 파일을 한 개씩 선택하세요."); return; }
  chooseFile(event.dataTransfer.files[0]);
});
for (const name of ["file", "url"]) $(name + "-tab").addEventListener("keydown", event => {
  if (["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) {
    event.preventDefault(); const next = event.key === "Home" ? "file" : event.key === "End" ? "url" : mode === "file" ? "url" : "file";
    tab(next); $(next + "-tab").focus();
  }
});
function frameCard(frame) {
  const card = document.createElement("article"); card.className = "frame";
  const button = document.createElement("button"); button.type = "button";
  button.setAttribute("aria-label", `${frame.index}번 프레임 ${frame.timestamp_sec.toFixed(3)}초 확대`);
  const image = document.createElement("img"); image.loading = "lazy";
  image.src = apiBase + frame.comparison_url; image.alt = `${frame.index}번 프레임 원본과 Grad-CAM`;
  button.append(image); card.append(button);
  const caption = document.createElement("div"); caption.className = "caption";
  const title = document.createElement("strong"); title.textContent = `${frame.index}번 · ${frame.timestamp_sec.toFixed(3)}초`;
  const text = document.createElement("p");
  text.textContent = `구간 ${frame.segment} · AI 점수 영향 ${frame.video_score_delta >= 0 ? "+" : ""}${frame.video_score_delta.toFixed(5)}\n강조 영역: ${frame.strongest_zone || "양의 신호 없음"}`;
  caption.append(title, text);
  if (frame.note) { const note = document.createElement("p"); note.className = "frame-note"; note.textContent = frame.note; caption.append(note); }
  card.append(caption);
  button.addEventListener("click", () => {
    $("zoom-title").textContent = `${frame.index}번 · ${frame.timestamp_sec.toFixed(3)}초 · 원본 / Grad-CAM`;
    $("zoom-image").src = apiBase + frame.comparison_url;
    $("zoom-note").textContent = `${frame.target_class === "fake" ? "AI 생성" : "실제 촬영"} 판단을 지지하는 양의 신호입니다. 합성 위치의 확정 표시가 아닙니다.`;
    $("zoom").showModal();
  });
  return card;
}
$("close-zoom").addEventListener("click", () => $("zoom").close());
function display(value) {
  result = value; detailsRendered = false;
  $("result-title").textContent = result.prediction_ko;
  $("score").textContent = result.fake_score.toFixed(4);
  $("result-meta").textContent = `${result.duration_sec.toFixed(1)}초 영상 · 분석 프레임 ${result.analyzed_frames}/64 · 분석 시간 ${result.elapsed_sec}초` +
    (result.skipped_segments.length ? `\n검은 화면 제외 후 미분석 구간 ${result.skipped_segments.length}개` : "");
  $("target-label").textContent = `강조 대상: ${result.prediction === "fake" ? "AI 생성" : "실제 촬영"}`;
  const byId = new Map(result.frames.map(frame => [frame.index, frame]));
  $("main-frames").replaceChildren(...result.main_frame_indices.map(id => frameCard(byId.get(id))));
  $("detail-frames").replaceChildren(); $("details").open = false;
  $("details-title").textContent = `세부사항 · 나머지 ${result.detail_frame_indices.length}개`;
  $("segments").replaceChildren(...result.segments.map(segment => {
    const node = document.createElement("span"); node.className = "segment";
    node.textContent = `${segment.start_sec.toFixed(1)}~${segment.end_sec.toFixed(1)}초 · ${segment.fake_score.toFixed(3)}`; return node;
  }));
  $("result").hidden = false;
  $("result").scrollIntoView({behavior:matchMedia("(prefers-reduced-motion: reduce)").matches ? "instant" : "smooth", block:"start"});
}
$("details").addEventListener("toggle", () => {
  if (!result || !$("details").open || detailsRendered) return;
  detailsRendered = true;
  const byId = new Map(result.frames.map(frame => [frame.index, frame]));
  $("detail-frames").replaceChildren(...result.detail_frame_indices.map(id => frameCard(byId.get(id))));
});
$("analysis-form").addEventListener("submit", async (event) => {
  event.preventDefault(); if (busy || !online) return;
  error("");
  let path, options;
  let timer;
  try {
    if (mode === "file") {
      const file = selectedFile;
      if (!file) throw new Error("분석할 영상 파일을 선택하세요.");
      if (!fileReady) throw new Error("지원하는 길이의 영상을 선택하거나 길이 확인이 끝날 때까지 기다려 주세요.");
      if (!file.size || file.size > 90 * 1024 ** 2) throw new Error("90 MB 이하의 영상 파일을 선택하세요.");
      path = "/jobs/file"; options = {method:"POST", headers:{"Content-Type":"application/octet-stream"}, body:file};
    } else {
      const url = new URL($("video-url").value.trim());
      if (url.protocol !== "https:" || !["youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be"].includes(url.hostname) || url.username || url.password) throw new Error("공개 YouTube Shorts 주소를 입력하세요.");
      path = "/jobs/youtube"; options = {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({url:url.href})};
    }
    busy = true; refreshButton(); $("progress").hidden = false; $("result").hidden = true;
    $("frame-progress").hidden = true; $("frame-progress").value = 0;
    const started = Date.now(); $("elapsed").textContent = "분석 준비 중 · 창을 닫지 말고 기다려 주세요.";
    timer = setInterval(() => { $("elapsed").textContent = `경과 ${Math.floor((Date.now() - started) / 1000)}초 · 대기 시간을 포함합니다.`; }, 1000);
    for (const id of ["video-file", "video-url", "remove-file", "file-tab", "url-tab"]) $(id).disabled = true;
    $("progress-title").textContent = "64프레임 분석"; status(mode === "file" ? "영상을 분석 서버로 전송하고 있습니다." : "공개 영상 주소를 확인하고 있습니다.");
    const job = await api(path, options);
    const deadline = Date.now() + 15 * 60 * 1000;
    while (Date.now() < deadline) {
      const state = await api("/jobs/" + encodeURIComponent(job.id));
      status(state.message);
      if (state.state === "failed") throw new Error(state.message);
      if (state.state === "completed") { display(state.result); return; }
      await sleep(2000);
    }
    throw new Error("응답 대기 시간이 지났습니다. 서버 상태를 확인해 주세요.");
  } catch (cause) {
    error(cause.message || "분석 서버에 연결하지 못했습니다. 잠시 후 다시 시도하세요.");
  } finally {
    clearInterval(timer); busy = false; $("progress").hidden = true;
    for (const id of ["video-file", "video-url", "remove-file", "file-tab", "url-tab"]) $(id).disabled = false;
    refreshButton();
  }
});
connect();
setInterval(() => { if (!document.hidden && !busy) connect(); }, 30000);
document.addEventListener("visibilitychange", () => { if (!document.hidden && !busy) connect(); });
