// 이미지 URL 을 직접 받아와(인스타 CDN CORS 우회) 확장 안의 모델(offscreen 문서)로 분석한다.
// 서버를 쓰지 않으며 이미지는 이 브라우저 밖으로 나가지 않는다.
const OFFSCREEN_URL = "offscreen.html";
const MAX_CONCURRENT = 1; // 모델은 한 번에 한 장씩 (wasm 1 스레드)

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

// ---------- offscreen 문서 ----------
let creating = null;
async function ensureOffscreen() {
  const contexts = await chrome.runtime.getContexts({
    contextTypes: ["OFFSCREEN_DOCUMENT"],
    documentUrls: [chrome.runtime.getURL(OFFSCREEN_URL)],
  });
  if (contexts.length) return;
  if (!creating) {
    creating = chrome.offscreen
      .createDocument({
        url: OFFSCREEN_URL,
        reasons: ["BLOBS"],
        justification: "얼굴 검출·판정 모델(WebAssembly)을 실행하고 이미지 Blob 을 디코딩합니다.",
      })
      .finally(() => {
        creating = null;
      });
  }
  await creating;
}

async function callModel(payload) {
  await ensureOffscreen();
  const response = await chrome.runtime.sendMessage({ target: "offscreen", ...payload });
  if (!response) throw new Error("모델 응답 없음");
  if (!response.ok) throw new Error(response.error);
  return response.result;
}

// ---------- 이미지 ----------
async function fetchImage(url) {
  const response = await fetch(url, { credentials: "include" });
  if (!response.ok) throw new Error(`이미지 다운로드 실패 (${response.status})`);
  return response.blob();
}

// 서비스 워커에는 FileReader 가 없으므로 직접 base64 로 바꾼다
async function blobToDataUrl(blob) {
  const bytes = new Uint8Array(await blob.arrayBuffer());
  let binary = "";
  for (let i = 0; i < bytes.length; i += 0x8000) binary += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return `data:${blob.type || "application/octet-stream"};base64,${btoa(binary)}`;
}

async function analyze({ url, dataUrl, endpoint = "predict", faceIndex }) {
  const imageDataUrl = dataUrl || (await blobToDataUrl(await fetchImage(url)));
  const result = await callModel({ type: "analyze", dataUrl: imageDataUrl, mode: "face", explain: endpoint === "explain" });
  if (endpoint === "explain" && faceIndex != null && result.faces.length) {
    if (faceIndex < 0 || faceIndex >= result.faces.length) throw new Error(`face_index 범위 밖: 0~${result.faces.length - 1}`);
    result.faces = [result.faces[faceIndex]];
    result.face_index = faceIndex;
  }
  return result;
}

async function health() {
  try {
    return await callModel({ type: "health" });
  } catch (error) {
    return { status: "error", error: String(error?.message || error) };
  }
}

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message?.target === "offscreen") return false; // offscreen 문서가 받는 메시지
  if (message?.type === "health") {
    health().then(sendResponse);
    return true;
  }
  if (message?.type === "predict" || message?.type === "explain") {
    const request = { ...message, endpoint: message.type };
    // 설명 요청은 사용자가 기다리는 중이므로 대기열 앞에 세우지 않고 바로 보낸다
    const run = message.type === "explain" ? analyze(request) : schedule(() => analyze(request));
    run
      .then((result) => sendResponse({ ok: true, result }))
      .catch((error) => sendResponse({ ok: false, error: String(error.message || error) }));
    return true;
  }
  return false;
});
