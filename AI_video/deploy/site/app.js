"use strict";
const $ = (id) => document.getElementById(id);
let mode = "file", apiBase = "", online = false, busy = false, detailsRendered = false, result = null;
const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));
function error(message) { $("error").textContent = message; $("error").hidden = !message; }
function status(message) { $("progress-message").textContent = message; }
async function api(path, options = {}) {
  const response = await fetch(apiBase + path, {cache:"no-store", ...options});
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "분석 서버에 연결하지 못했습니다.");
  return data;
}
async function connect() {
  try {
    const response = await fetch("/api/config", {cache:"no-store"});
    if (!response.ok) throw new Error("연결 설정 확인 실패");
    const config = await response.json();
    if (!config.configured) throw new Error("분석 서버 연결 준비 중");
    apiBase = config.apiBase || "";
    await api("/health", {signal:AbortSignal.timeout(12000)});
    online = true;
    $("connection").textContent = "● 분석 서버 연결됨";
    $("connection").className = "status online";
  } catch (cause) {
    online = false;
    $("connection").textContent = "● 분석 서버 오프라인";
    $("connection").className = "status offline";
    $("availability").textContent = "분석 서버가 연결되지 않았습니다. 운영자가 노트북 분석 서버를 실행하면 이용할 수 있습니다.";
  }
  $("analyze").disabled = !online || busy;
}
function tab(next) {
  mode = next;
  for (const name of ["file", "url"]) {
    $(name + "-tab").setAttribute("aria-selected", String(name === mode));
    $(name + "-panel").hidden = name !== mode;
  }
  error("");
}
$("file-tab").addEventListener("click", () => tab("file"));
$("url-tab").addEventListener("click", () => tab("url"));
$("video-file").addEventListener("change", () => {
  const file = $("video-file").files[0];
  $("filename").textContent = file ? `${file.name} · ${(file.size / 1024 ** 2).toFixed(1)} MB` : "선택한 파일이 없습니다.";
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
  try {
    if (mode === "file") {
      const file = $("video-file").files[0];
      if (!file) throw new Error("분석할 영상 파일을 선택하세요.");
      if (!file.size || file.size > 90 * 1024 ** 2) throw new Error("90 MB 이하의 영상 파일을 선택하세요.");
      path = "/jobs/file"; options = {method:"POST", headers:{"Content-Type":"application/octet-stream"}, body:file};
    } else {
      const url = new URL($("video-url").value.trim());
      if (url.protocol !== "https:" || !["youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be"].includes(url.hostname) || url.username || url.password) throw new Error("공개 YouTube Shorts 주소를 입력하세요.");
      path = "/jobs/youtube"; options = {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({url:url.href})};
    }
    busy = true; $("analyze").disabled = true; $("progress").hidden = false; $("result").hidden = true;
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
    busy = false; $("progress").hidden = true; $("analyze").disabled = !online;
  }
});
connect();
