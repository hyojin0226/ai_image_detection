// 이미지 URL 을 직접 받아와(인스타 CDN CORS 우회) 로컬 서버 /predict 로 보낸다.
const SERVER = "http://127.0.0.1:8000";
const MAX_CONCURRENT = 2; // 노트북 CPU 부담을 줄이기 위해 동시에 2장만

let running = 0;
const queue = [];

function schedule(task) {
  return new Promise((resolve, reject) => {
    queue.push({ task, resolve, reject });
    pump();
  });
}

function pump() {
  while (running < MAX_CONCURRENT && queue.length) {
    const { task, resolve, reject } = queue.shift();
    running++;
    task()
      .then(resolve, reject)
      .finally(() => {
        running--;
        pump();
      });
  }
}

async function fetchImage(url) {
  const response = await fetch(url, { credentials: "include" });
  if (!response.ok) throw new Error(`이미지 다운로드 실패 (${response.status})`);
  return response.blob();
}

function dataUrlToBlob(dataUrl) {
  const [header, data] = dataUrl.split(",", 2);
  const mime = (header.match(/data:([^;]+)/) || [])[1] || "application/octet-stream";
  const bytes = header.includes(";base64") ? Uint8Array.from(atob(data), (c) => c.charCodeAt(0)) : new TextEncoder().encode(decodeURIComponent(data));
  return new Blob([bytes], { type: mime });
}

async function predict({ url, dataUrl, endpoint = "predict", faceIndex }) {
  const blob = dataUrl ? dataUrlToBlob(dataUrl) : await fetchImage(url);
  const form = new FormData();
  form.append("file", blob, "image");
  const query = faceIndex == null ? "" : `?face_index=${faceIndex}`;
  let response;
  try {
    response = await fetch(`${SERVER}/${endpoint}${query}`, { method: "POST", body: form });
  } catch (error) {
    throw new Error("SERVER_DOWN");
  }
  if (!response.ok) {
    const detail = await response.text().catch(() => "");
    throw new Error(`서버 오류 ${response.status}: ${detail.slice(0, 200)}`);
  }
  return response.json();
}

async function health() {
  try {
    const response = await fetch(`${SERVER}/health`);
    return response.ok ? await response.json() : { status: "error" };
  } catch (error) {
    return { status: "down" };
  }
}

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message?.type === "health") {
    health().then(sendResponse);
    return true;
  }
  if (message?.type === "predict" || message?.type === "explain") {
    const request = { ...message, endpoint: message.type };
    // 설명 요청은 사용자가 기다리는 중이므로 대기열 앞에 세우지 않고 바로 보낸다
    const run = message.type === "explain" ? predict(request) : schedule(() => predict(request));
    run
      .then((result) => sendResponse({ ok: true, result }))
      .catch((error) => sendResponse({ ok: false, error: String(error.message || error) }));
    return true;
  }
  return false;
});
