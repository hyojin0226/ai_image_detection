// Detection: final ConvNeXt-T face model (ONNX) run in the browser.
// BlazeFace finds faces, each face is cropped at 2x its box, resized to 256x256 and classified.
// The largest face decides the image verdict (same rule as the develop-branch server).
// Images are queued and analysed one at a time; results are kept in a local history.
import { classifyFile, supportedImageMime } from './file-types.js';
import { metric } from './inference-client.js?v=classifiers';
import { loadImageEngine } from './image-engine.js?v=photo-stages';
const MAX_QUEUE = 20;                 // images per batch
const MAX_BYTES = 15 * 1024 * 1024;   // per image

const $ = (id) => document.getElementById(id);
const els = {
  status: $("model-status"), statusText: $("status-text"), loadFill: $("loadbar-fill"), modelError: $("model-error"),
  dropzone: $("dropzone"), input: $("file-input"), inputMore: $("file-input-more"), pasteBtn: $("paste-btn"),
  error: $("error"), queueSection: $("queue-section"), queue: $("queue"),
  summaryMain: $("summary-main"), summarySub: $("summary-sub"), addMore: $("add-more"), clearQueue: $("clear-queue"),
  thresholdText: $("threshold-text"),
  recentStrip: $("recent-strip"), recentList: $("recent-list"),
  historyList: $("history-list"), historyEmpty: $("history-empty"), historyCount: $("history-count"),
  historyToggle: $("history-toggle"), historyClear: $("history-clear"), dockCount: $("dock-history-count"),
  detail: $("detail-modal"), detailImg: $("detail-img"), detailVerdict: $("detail-verdict"), detailSub: $("detail-sub"),
  detailRealBar: $("detail-real-bar"), detailFakeBar: $("detail-fake-bar"), detailThreshold: $("detail-threshold"),
  detailReal: $("detail-real"), detailFake: $("detail-fake"), detailThresholdText: $("detail-threshold-text"),
  detailName: $("detail-name"), detailClose: $("detail-close"),
};
const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;
const { toast, go, routeName, openModal, closeModal } = window.Fakit;

let config = null;
let engine = null;
let modelReady = null;   // promise: resolves when the model can predict, rejects if loading failed
let photoInFlight = null;
const state = { items: [], running: false, nextId: 1, filter: "all" };

// ── Model ──────────────────────────────────────────────────────────────
function loadResources() {
  if (modelReady) return modelReady;
  metric('model-load-start');
  els.status.classList.remove('failed'); els.modelError.hidden=true;
  modelReady = loadImageEngine({onProgress:progress=>metric('model-progress',{progress})}).then(result => {
    engine=result;config=result.config;
    els.status.classList.add('ready');els.statusText.textContent='분석 준비 완료';
    metric('model-ready',{backend:'onnxruntime-web wasm'});
    return result;
  }).catch(error=>{
    modelReady=null;els.status.classList.add('failed');els.statusText.textContent='모델을 불러오지 못했습니다';
    els.modelError.textContent='연결을 확인하고 다시 검사해 주세요.';els.modelError.hidden=false;throw error;
  });
  modelReady.catch(()=>{}); return modelReady;
}
document.addEventListener('fakit:metric',e=>{
  if(e.detail.name==='model-progress')els.loadFill.style.width=Math.round(e.detail.data.progress*100)+'%';
});
const threshold = () => Number(config.decision_threshold_real.face);

// requestAnimationFrame stalls in background tabs; never let that hold up a result.
const nextFrame = () => new Promise((resolve) => {
  const fallback = setTimeout(resolve, 100);
  requestAnimationFrame(() => { clearTimeout(fallback); resolve(); });
});
const pct = (p) => `${(p * 100).toFixed(1)}%`;
const escapeHTML = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

// ── Adding images ──────────────────────────────────────────────────────
function showError(msg) {
  els.error.textContent = msg;
  els.error.hidden = false;
}

// Pasted images skip the JPG/PNG/WEBP check: capture tools put BMP/GIF/etc. on the clipboard, and
// anything the browser can't decode is still rejected by decode() below.
function acceptable(file, pasted) {
  return pasted ? file.type.startsWith("image/") : Boolean(supportedImageMime(file));
}

async function addFiles(fileList, { pasted = false } = {}) {
  const files = [...fileList];
  if (!files.length) return;
  els.error.hidden = true;
  if (routeName() !== "home") go("home");

  const problems = [];
  const accepted = [];
  for (const file of files) {
    const kind = classifyFile(file);
    if (kind === "video") { problems.push(`${file.name}: 영상 파일입니다. 영상 분석은 아직 지원하지 않습니다.`); continue; }
    if (kind === "text") { problems.push(`${file.name}: 글·문서 파일입니다. 글 분석은 아직 지원하지 않습니다.`); continue; }
    if (state.items.length + accepted.length >= MAX_QUEUE) { problems.push(`한 번에 최대 ${MAX_QUEUE}장까지 올릴 수 있습니다.`); break; }
    if (!acceptable(file, pasted)) { problems.push(`${file.name || "이미지"}: JPG, PNG, WEBP만 지원합니다.`); continue; }
    if (file.size > MAX_BYTES) { problems.push(`${file.name}: 15MB를 넘습니다.`); continue; }
    accepted.push(file);
  }

  for (const [i, file] of accepted.entries()) {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.src = url;
    try {
      await img.decode();
    } catch {
      URL.revokeObjectURL(url);
      problems.push(`${file.name || "이미지"}: 이미지 파일을 읽을 수 없습니다.`);
      continue;
    }
    const name = pasted && (!file.name || file.name === "clipboard")
      ? `붙여넣은 이미지 ${new Date().toLocaleTimeString("ko-KR", { hour: "2-digit", minute: "2-digit" })}`
      : file.name;
    const item = { id: state.nextId++, name, url, img, file, status: "queued", real: null, fake: null, verdict: null, pasted, at: Date.now() };
    state.items.push(item);
    mountItem(item);
    if (i === 0) els.queueSection.scrollIntoView({ behavior: reduceMotion ? "instant" : "smooth", block: "nearest" });
  }

  renderSummary();
  if (problems.length) showError(problems.slice(0, 3).join(" "));
  pump();
}

// ── Queue processing ───────────────────────────────────────────────────
async function pump() {
  if (state.running) return;
  state.running = true;
  try {
    while (true) {
      const item = state.items.find((i) => i.status === "queued");
      if (!item) break;
      item.status = "analyzing";
      renderItem(item);
      renderProcess(item, "waiting");
      try {
        await loadResources();
      } catch {
        state.items.filter((i) => i.status !== "done").forEach((i) => { i.status = "error"; i.error = "모델 없음"; renderItem(i); });
        renderProcess(item, "error");
        break;
      }
      await analyzeItem(item);
      renderSummary();
    }
  } finally {
    state.running = false;
    renderSummary();
  }
}

async function analyzeItem(item, { signal, onStage } = {}) {
  if (!state.items.includes(item)) return;
  renderProcess(item, "analyzing");
  await nextFrame();
  try {
    if (photoInFlight) throw new Error('이전 사진 검사가 아직 끝나지 않았어요. 잠시 기다리거나 페이지를 새로고침해 주세요.');
    const deadline = AbortSignal.timeout(90000);
    const operationSignal = signal ? AbortSignal.any([signal, deadline]) : deadline;
    operationSignal.throwIfAborted();
    const task = engine.analyze(item.file, { mode: 'face', explain: true, ela: false, signal: operationSignal, onStage });
    photoInFlight = task;
    const clear = () => { if (photoInFlight === task) photoInFlight = null; };
    task.then(clear, clear);
    let onAbort;
    const stopped = new Promise((_, reject) => {
      onAbort = () => reject(deadline.aborted
        ? new Error('사진 검사 시간이 초과됐어요. 페이지를 새로고침하고 한 장씩 다시 검사해 주세요.')
        : operationSignal.reason);
      if (operationSignal.aborted) onAbort();
      else operationSignal.addEventListener('abort', onAbort, { once: true });
    });
    let result;
    try {
      result = await Promise.race([task, stopped]);
    } finally {
      operationSignal.removeEventListener('abort', onAbort);
    }
    if (!state.items.includes(item)) return;   // removed while analysing
    if (result.status === 'no_face') {
      item.status = "error"; item.error = "판단 불가 · 얼굴 없음"; item.noFace = true;
      renderItem(item); renderProcess(item, "error"); return;
    }
    const face = result.faces[0];
    const probabilityReal = face.prob_real_raw;
    if(!Number.isFinite(probabilityReal)||probabilityReal<0||probabilityReal>1)throw Error('잘못된 모델 출력');
    item.faces = result.faces;
    item.real = probabilityReal;
    item.fake = 1 - probabilityReal;
    item.verdict = probabilityReal >= threshold() ? "real" : "fake";
    item.status = "done";
    renderItem(item);
    renderProcess(item, "done");
  } catch (err) {
    console.error(err);
    if (!state.items.includes(item)) return;
    item.status = "error";
    item.error = "분석 실패";
    item.failure = err;
    renderItem(item);
    renderProcess(item, "error");
  }
}

// ── Queue rendering ────────────────────────────────────────────────────
function cardHTML(entry, { removable = true, date = false } = {}) {
  return `
    ${removable ? `<button type="button" class="shot-remove" aria-label="목록에서 제거">×</button>` : ""}
    <div class="shot-img"><img src="${entry.url || entry.thumb}" alt=""><span class="shot-chip"></span></div>
    <div class="shot-meta"><span class="shot-verdict"></span><span class="shot-prob"></span></div>
    <p class="shot-name" title="${escapeHTML(entry.name)}">${escapeHTML(entry.name)}</p>
    ${date ? `<p class="shot-date">${formatDate(entry.at)}</p>` : ""}`;
}
function applyStatus(el, entry) {
  el.dataset.status = entry.status || "done";
  if (entry.verdict) el.dataset.verdict = entry.verdict; else delete el.dataset.verdict;
  const chip = el.querySelector(".shot-chip");
  const verdict = el.querySelector(".shot-verdict");
  const prob = el.querySelector(".shot-prob");
  if (entry.status === "queued") { chip.textContent = "대기"; verdict.textContent = "대기 중"; prob.textContent = ""; }
  else if (entry.status === "analyzing") { chip.textContent = "분석 중"; verdict.textContent = "분석 중…"; prob.textContent = ""; }
  else if (entry.status === "error") { chip.textContent = "오류"; verdict.textContent = entry.error || "분석 실패"; prob.textContent = ""; }
  else {
    const isReal = entry.verdict === "real";
    chip.textContent = isReal ? "REAL" : "AI";
    verdict.textContent = isReal ? "REAL" : "AI GENERATED";
    prob.textContent = `REAL 점수 ${entry.real.toFixed(3)}`;
  }
}
function mountItem(item) {
  const el = document.createElement("article");
  el.className = "shot-card";
  el.tabIndex = 0;
  el.setAttribute("role", "button");
  el.innerHTML = cardHTML(item);
  el.querySelector(".shot-remove").addEventListener("click", (e) => { e.stopPropagation(); removeItem(item); });
  const open = () => { if (item.status === "done") openDetail(item); };
  el.addEventListener("click", open);
  el.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); open(); } });
  item.el = el;
  applyStatus(el, item);
  els.queue.append(el);
  els.queueSection.hidden = false;
}
function renderItem(item) {
  if (item.el) applyStatus(item.el, item);
}
function renderProcess(item, phase) {
  if (!state.items.includes(item)) return;
  const view = $("process-view");
  document.dispatchEvent(new CustomEvent('fakit:inspect', { detail: { url: item.url, state: phase, label: phase === 'done' ? `${item.verdict === 'real' ? 'REAL' : 'AI'} · REAL 점수 ${item.real.toFixed(3)}` : phase === 'error' ? (item.noFace ? '판단 불가 · 얼굴 없음' : '분석 실패') : phase === 'waiting' ? '모델 준비 중' : '분석 중' } }));
  view.hidden = false;
  view.dataset.phase = phase;
  view.dataset.item = String(item.id);
  $("process-name").textContent = item.name;
  $("process-original").src = item.url;
  $("process-size").textContent = `${item.img.naturalWidth} × ${item.img.naturalHeight}`;
  const input = $("process-input");
  const crop = item.faces?.[0]?.crop_image;
  input.hidden = !crop;
  if (crop) {
    const image = new Image();
    image.onload = () => { input.width = image.width; input.height = image.height; input.getContext("2d").drawImage(image, 0, 0); };
    image.src = crop;
    $("process-input-size").textContent = `얼굴 crop ${config.input_size} × ${config.input_size}`;
  }
  const faceCount = item.faces?.length > 1 ? ` · 얼굴 ${item.faces.length}개 중 가장 큰 얼굴 기준` : "";
  const messages = { waiting: "모델 준비 대기", analyzing: "얼굴 검출 · 얼굴 crop 분석 중", done: "분석 완료" + faceCount, error: item.noFace ? "얼굴을 찾지 못해 판단할 수 없습니다" : "분석 실패 · 새로고침 후 다시 시도하세요" };
  $("process-status").textContent = messages[phase];
  $("process-score").hidden = phase !== "done";
  $("process-verdict").textContent = phase === "done" ? (item.verdict === "real" ? "REAL 분류" : "AI 분류") : phase === "error" ? "결과 없음" : "결과 대기";
  if (phase === "done") {
    $("process-boundary").style.left = `${threshold() * 100}%`;
    $("process-point").style.left = `${item.real * 100}%`;
    $("process-value").textContent = `REAL 점수 ${item.real.toFixed(3)} · 기준 ${threshold().toFixed(2)}`;
  }
}
function removeItem(item) {
  const idx = state.items.indexOf(item);
  if (idx < 0) return;
  state.items.splice(idx, 1);
  item.el.remove();
  URL.revokeObjectURL(item.url);
  renderSummary();
  if ($("process-view").dataset.item === String(item.id)) $("process-view").hidden = true;
}
function clearQueue() {
  document.dispatchEvent(new CustomEvent('fakit:inspect', { detail: { state: 'idle', label: 'DROP A FILE' } }));
  state.items.forEach((i) => { i.el.remove(); URL.revokeObjectURL(i.url); });
  state.items.length = 0;
  renderSummary();
  $("process-view").hidden = true;
}
function renderSummary() {
  const n = state.items.length;
  els.queueSection.hidden = n === 0;
  if (!n) return;
  const real = state.items.filter((i) => i.verdict === "real").length;
  const fake = state.items.filter((i) => i.verdict === "fake").length;
  const pending = state.items.filter((i) => i.status === "queued" || i.status === "analyzing").length;
  const failed = state.items.filter((i) => i.status === "error").length;
  const done = state.items.filter(i => i.status === "done").length;
  els.summaryMain.textContent = `${n}장 중 ${done}장 분석 완료`;
  const parts = [];
  if (real) parts.push(`<b class="real">REAL ${real}장</b>`);
  if (fake) parts.push(`<b class="fake">AI ${fake}장</b>`);
  if (pending) parts.push(`분석 중 ${pending}장`);
  if (failed) parts.push(`실패 ${failed}장`);
  els.summarySub.innerHTML = parts.join(" · ") || "분석을 시작합니다…";
  els.addMore.disabled = n >= MAX_QUEUE;
}

// ── Detail modal (works for queue items and history entries) ───────────
function openDetail(entry) {
  const isReal = entry.verdict === "real";
  const t = entry.threshold ?? threshold();
  els.detailImg.src = entry.url || entry.thumb;
  els.detailVerdict.textContent = isReal ? "REAL" : "AI GENERATED";
  els.detailVerdict.className = `prediction ${isReal ? "is-real" : "is-fake"}`;
  els.detailSub.textContent = isReal ? "모델이 실제 사진 범주로 분류했습니다. 진위가 확인된 것은 아닙니다." : "모델이 AI 생성 범주로 분류했습니다. 생성 사실을 입증하는 결과는 아닙니다.";
  els.detailReal.textContent = pct(entry.real);
  els.detailFake.textContent = pct(entry.fake);
  els.detailThreshold.style.left = `${t * 100}%`;
  els.detailThresholdText.textContent = `REAL 점수 ${entry.real.toFixed(3)} / 기준선 ${t.toFixed(2)}. 점수는 모델 출력이며, 이 사진에 대한 정답률이 아닙니다.`;
  els.detailName.textContent = entry.at ? `${entry.name} · ${formatDate(entry.at)}` : entry.name;
  els.detailRealBar.style.width = "0";
  els.detailFakeBar.style.width = "0";
  openModal(els.detail);
  // Widths were reset above; set them a frame later so the bars animate.
  requestAnimationFrame(() => requestAnimationFrame(() => {
    els.detailRealBar.style.width = `calc(${entry.real * 100}% - 1px)`;
    els.detailFakeBar.style.width = `calc(${entry.fake * 100}% - 1px)`;
  }));
}
els.detailClose.addEventListener("click", () => closeModal(els.detail));
$("detail-help").addEventListener("click", () => closeModal(els.detail));

function formatDate(ts) { return new Date(ts).toLocaleString("ko-KR"); }

// Unified UI bridge. Model preprocessing, predict, threshold and history are unchanged.
window.Fakit.analyzeImage = async (file, { signal, onState = () => {} } = {}) => {
  const started = performance.now();
  signal?.throwIfAborted();
  onState('loading', '이미지 분석을 준비하고 있어요.');
  await loadResources();
  signal?.throwIfAborted();
  const url = URL.createObjectURL(file);
  const img = new Image();
  let item;
  const cancel = () => { if (item && state.items.includes(item)) removeItem(item); };
  try {
    img.src = url;
    await img.decode();
    signal?.throwIfAborted();
    const thumbCanvas = document.createElement('canvas');
    const scale = Math.min(1, 160 / Math.max(img.naturalWidth, img.naturalHeight));
    thumbCanvas.width = Math.max(1, Math.round(img.naturalWidth * scale));
    thumbCanvas.height = Math.max(1, Math.round(img.naturalHeight * scale));
    const thumbContext = thumbCanvas.getContext('2d');
    thumbContext.drawImage(img, 0, 0, thumbCanvas.width, thumbCanvas.height);
    const thumb = thumbCanvas.toDataURL('image/jpeg', .72);
    item = { id: state.nextId++, name: file.name, url, img, file, status: 'analyzing', real: null, fake: null, verdict: null, at: Date.now() };
    state.items.push(item);
    mountItem(item);
    signal?.addEventListener('abort', cancel, { once: true });
    onState('analyzing', '이미지를 검사하고 있어요.');
    await analyzeItem(item, { signal, onStage: message => onState('analyzing', message) });
    signal?.throwIfAborted();
    if (item.noFace) throw new Error('사진에서 얼굴을 찾지 못해 판단할 수 없어요. 얼굴이 잘 보이는 사진을 올려 주세요.');
    if (item.status !== 'done') throw item.failure || new Error('이미지를 분석하지 못했어요. 다시 시도해 주세요.');
    // Grad-CAM comes from the same forward pass (closed-form head gradient in the ONNX graph).
    const face = item.faces[0];
    let explanation, explanationError;
    if (face.crop_image && face.gradcam_image && face.gradcam_heatmap) {
      explanation = { original: face.crop_image, heatmap: face.gradcam_heatmap, image: face.gradcam_image, target: face.gradcam_target, layer: 'convnext stage 4 (8 × 8)' };
    } else {
      explanationError = '판별 점수는 계산했지만 이 환경에서는 Grad-CAM을 표시하지 못했어요.';
    }
    const faces = item.faces.map(f => ({ real: f.prob_real, verdict: f.label, bbox: f.bbox }));
    signal?.throwIfAborted();
    metric('image-result');
    return { status: 'result', real: item.real, fake: item.fake, verdict: item.verdict, threshold: threshold(), model: config.model_name, thumb, faces, explanation, explanationError, elapsedMs: performance.now() - started };
  } finally {
    signal?.removeEventListener('abort', cancel);
    cancel();
    URL.revokeObjectURL(url);
    img.removeAttribute('src');
    $('process-original').removeAttribute('src');
    const canvas = $('process-input'); canvas.width = canvas.height = 1;
  }
};
document.addEventListener('fakit:inference-released', () => {
  modelReady = null;
  els.status.classList.remove('ready');
});
metric("ui-ready");
