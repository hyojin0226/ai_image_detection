// Detection: same model, preprocessing and threshold as streamlit_app.py, run in the browser with
// TF.js. Images are queued and analysed one at a time; results are kept in a local history.
import { classifyFile, supportedImageMime } from './file-types.js';
import { computeGradcam, overlayGradcam } from './gradcam.js';
const MODEL_URL = "model/model.json";
const CONFIG_URL = "model/inference_config.json";
const MAX_QUEUE = 20;                 // images per batch
const MAX_BYTES = 15 * 1024 * 1024;   // per image
const THUMB_SIZE = 160;               // history stores a small JPEG thumbnail, never the original
const HISTORY_KEY = "fakit.history.v1";
const HISTORY_ON_KEY = "fakit.history.enabled";
const HISTORY_MAX = 200;

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

let model = null;
let config = null;
let modelReady = null;   // promise: resolves when the model can predict, rejects if loading failed
const state = { items: [], running: false, nextId: 1, filter: "all" };

// ── Model ──────────────────────────────────────────────────────────────
function loadResources() {
  modelReady = (async () => {
    try {
      config = await (await fetch(CONFIG_URL)).json();
      model = await tf.loadGraphModel(MODEL_URL, {
        onProgress: (p) => {
          const percent = Math.round(p * 100);
          els.loadFill.style.width = `${percent}%`;
          els.statusText.textContent = `모델 불러오는 중… ${percent}%`;
        },
      });
      // Warm-up so the first real prediction isn't slowed by shader compilation.
      const [h, w] = config.image_size;
      tf.tidy(() => model.predict(tf.zeros([1, h, w, 3])));
      els.status.classList.add("ready");
      els.statusText.textContent = `모델 준비 완료 (${tf.getBackend()})`;
      els.thresholdText.textContent = `REAL 확률이 기준선 ${threshold().toFixed(2)} 이상이면 REAL로 판정합니다. 결과는 참고용입니다.`;
    } catch (err) {
      console.error(err);
      els.status.classList.add("failed");
      els.statusText.textContent = "모델을 불러오지 못했습니다";
      els.modelError.textContent = "네트워크 연결을 확인한 뒤 새로고침해 주세요.";
      els.modelError.hidden = false;
      throw err;
    }
  })();
  modelReady.catch(() => {});
  return modelReady;
}
const threshold = () => Number(config.decision_threshold_real);

// Equivalent of PIL image.convert("RGB").resize(image_size) → float32 (0–255; the model's own
// Rescaling layer normalises).
function inputCanvas(img) {
  const [h, w] = config.image_size;
  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.fillStyle = "#000";                 // flatten transparency like convert("RGB")
  ctx.fillRect(0, 0, w, h);
  ctx.drawImage(img, 0, 0, w, h);
  return canvas;
}
function toInputTensor(img) { return tf.tidy(() => tf.browser.fromPixels(inputCanvas(img), 3).toFloat().expandDims(0)); }

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
    const item = { id: state.nextId++, name, url, img, status: "queued", real: null, fake: null, verdict: null, pasted, at: Date.now() };
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
        await modelReady;
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

async function analyzeItem(item) {
  if (!state.items.includes(item)) return;
  renderProcess(item, "analyzing");
  await nextFrame();
  try {
    const input = toInputTensor(item.img);
    const output = model.predict(input);
    const probabilityReal = (await output.data())[0];
    tf.dispose([input, output]);
    if (!state.items.includes(item)) return;   // removed while analysing
    item.real = probabilityReal;
    item.fake = 1 - probabilityReal;
    item.verdict = probabilityReal >= threshold() ? "real" : "fake";
    item.status = "done";
    renderItem(item);
    renderProcess(item, "done");
    saveToHistory(item);
  } catch (err) {
    console.error(err);
    if (!state.items.includes(item)) return;
    item.status = "error";
    item.error = "분석 실패";
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
  document.dispatchEvent(new CustomEvent('fakit:inspect', { detail: { url: item.url, state: phase, label: phase === 'done' ? `${item.verdict === 'real' ? 'REAL' : 'AI'} · REAL 점수 ${item.real.toFixed(3)}` : phase === 'error' ? '분석 실패' : phase === 'waiting' ? '모델 준비 중' : '분석 중' } }));
  view.hidden = false;
  view.dataset.phase = phase;
  view.dataset.item = String(item.id);
  $("process-name").textContent = item.name;
  $("process-original").src = item.url;
  $("process-size").textContent = `${item.img.naturalWidth} × ${item.img.naturalHeight}`;
  const input = $("process-input");
  input.hidden = !config;
  if (config) {
    const prepared = inputCanvas(item.img);
    input.width = prepared.width; input.height = prepared.height;
    input.getContext("2d").drawImage(prepared, 0, 0);
    $("process-input-size").textContent = `${prepared.width} × ${prepared.height}`;
  }
  const messages = { waiting: "모델 준비 대기", analyzing: "이미지 전체 분석 중", done: "분석 완료", error: "분석 실패 · 새로고침 후 다시 시도하세요" };
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

// ── History (localStorage, thumbnails only) ────────────────────────────
function historyEnabled() {
  try { return localStorage.getItem(HISTORY_ON_KEY) !== "0"; } catch { return false; }
}
function loadHistory() {
  try { return JSON.parse(localStorage.getItem(HISTORY_KEY)) || []; } catch { return []; }
}
function storeHistory(list) {
  try {
    localStorage.setItem(HISTORY_KEY, JSON.stringify(list));
    return true;
  } catch {
    toast("기록을 저장할 공간이 부족합니다. 오래된 기록을 지워 주세요.");
    return false;
  }
}
function makeThumb(img) {
  const w = img.naturalWidth, h = img.naturalHeight;
  const scale = Math.min(1, THUMB_SIZE / Math.max(w, h));
  const canvas = document.createElement("canvas");
  canvas.width = Math.max(1, Math.round(w * scale));
  canvas.height = Math.max(1, Math.round(h * scale));
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL("image/jpeg", 0.72);
}
function saveToHistory(item) {
  if (!historyEnabled()) return;
  const list = loadHistory();
  list.unshift({
    id: `${item.at}-${item.id}`, name: item.name, thumb: makeThumb(item.img),
    verdict: item.verdict, real: item.real, fake: item.fake, threshold: threshold(), at: Date.now(),
  });
  while (list.length > HISTORY_MAX) list.pop();
  if (storeHistory(list)) refreshHistoryViews();
}
function deleteHistoryEntry(id) {
  const list = loadHistory().filter((e) => e.id !== id);
  storeHistory(list);
  refreshHistoryViews();
}
function formatDate(ts) {
  return new Date(ts).toLocaleString("ko-KR", { year: "numeric", month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit" });
}

function renderHistoryPage() {
  const all = loadHistory();
  const list = state.filter === "all" ? all : all.filter((e) => e.verdict === state.filter);
  els.historyCount.textContent = all.length ? `${all.length}건 저장됨` : "";
  els.historyClear.hidden = all.length === 0;
  els.historyEmpty.hidden = list.length > 0;
  els.historyList.innerHTML = "";
  if (!list.length && all.length) {
    els.historyEmpty.hidden = false;
    els.historyEmpty.querySelector("strong").textContent = state.filter === "real" ? "REAL로 판정한 기록이 없습니다." : "AI로 판정한 기록이 없습니다.";
  } else {
    els.historyEmpty.querySelector("strong").textContent = "아직 기록이 없습니다.";
  }
  list.forEach((entry, i) => {
    const el = document.createElement("article");
    el.className = "shot-card";
    el.tabIndex = 0;
    el.setAttribute("role", "button");
    el.style.animationDelay = `${Math.min(i, 12) * 30}ms`;
    el.innerHTML = cardHTML(entry, { date: true });
    applyStatus(el, entry);
    el.querySelector(".shot-remove").addEventListener("click", (e) => { e.stopPropagation(); deleteHistoryEntry(entry.id); });
    el.addEventListener("click", () => openDetail(entry));
    el.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); openDetail(entry); } });
    els.historyList.append(el);
  });
}
function renderRecentStrip() {
  const list = loadHistory().slice(0, 8);
  els.recentStrip.hidden = list.length === 0;
  els.recentList.innerHTML = "";
  list.forEach((entry) => {
    const el = document.createElement("button");
    el.type = "button";
    el.className = "recent-item";
    el.dataset.verdict = entry.verdict;
    el.title = `${entry.name} · ${entry.verdict === "real" ? "REAL" : "AI"} ${pct(entry.verdict === "real" ? entry.real : entry.fake)}`;
    const thumb = document.createElement("img");
    thumb.src = entry.thumb;
    thumb.alt = "";
    const label = document.createElement("b");
    label.textContent = entry.verdict === "real" ? "REAL" : "AI";
    const name = document.createElement("span");
    name.className = "recent-name";
    name.textContent = entry.name;
    const score = document.createElement("span");
    score.className = "recent-score";
    score.textContent = `REAL 점수 ${pct(entry.real)}`;
    el.setAttribute("aria-label", `${entry.name}, ${label.textContent}, ${score.textContent}, 상세 결과 보기`);
    el.append(thumb, label, name, score);
    el.addEventListener("click", () => openDetail(entry));
    els.recentList.append(el);
  });
}
function refreshHistoryViews() {
  const count = loadHistory().length;
  els.dockCount.textContent = count > 99 ? "99+" : String(count);
  els.dockCount.hidden = count === 0;
  renderRecentStrip();
  if (routeName() === "history") renderHistoryPage();
}
els.historyToggle.checked = historyEnabled();
els.historyToggle.addEventListener("change", () => {
  try { localStorage.setItem(HISTORY_ON_KEY, els.historyToggle.checked ? "1" : "0"); } catch {}
  toast(els.historyToggle.checked ? "이제부터 결과를 기록에 저장합니다." : "기록 저장을 껐습니다. 기존 기록은 남아 있습니다.");
});
els.historyClear.addEventListener("click", () => {
  if (!confirm("저장된 기록을 모두 삭제할까요? 되돌릴 수 없습니다.")) return;
  storeHistory([]);
  refreshHistoryViews();
  toast("기록을 모두 삭제했습니다.");
});
document.querySelectorAll(".chip[data-filter]").forEach((chip) => chip.addEventListener("click", () => {
  state.filter = chip.dataset.filter;
  document.querySelectorAll(".chip[data-filter]").forEach((c) => c.classList.toggle("is-on", c === chip));
  renderHistoryPage();
}));
document.addEventListener("fakit:route", (e) => {
  if (e.detail.route === "history") renderHistoryPage();
});

// Unified UI bridge. Model preprocessing, predict, threshold and history are unchanged.
window.Fakit.analyzeImage = async (file, { signal, onState = () => {} } = {}) => {
  signal?.throwIfAborted();
  onState('loading', '이미지 분석을 준비하고 있어요.');
  if (els.status.classList.contains('failed')) { els.status.classList.remove('failed'); loadResources(); }
  await modelReady;
  signal?.throwIfAborted();
  const url = URL.createObjectURL(file);
  const img = new Image();
  let item;
  const cancel = () => { if (item && state.items.includes(item)) removeItem(item); };
  try {
    img.src = url;
    await img.decode();
    signal?.throwIfAborted();
    item = { id: state.nextId++, name: file.name, url, img, status: 'analyzing', real: null, fake: null, verdict: null, at: Date.now() };
    state.items.push(item);
    mountItem(item);
    signal?.addEventListener('abort', cancel, { once: true });
    onState('analyzing', '이미지를 검사하고 있어요.');
    await analyzeItem(item);
    signal?.throwIfAborted();
    if (item.status !== 'done') throw new Error('이미지를 분석하지 못했어요. 다시 시도해 주세요.');
    let explanation, explanationError;
    onState('analyzing', 'AI 판정에 영향을 준 영역을 계산하고 있어요.');
    await nextFrame();
    let explainInput;
    try {
      explainInput = toInputTensor(img);
      const map = await computeGradcam(tf, model, explainInput, item.real, { signal });
      explanation = { image: overlayGradcam(img, map), target: map.target, layer: map.layer };
    } catch (error) {
      signal?.throwIfAborted();
      explanationError = '판별 점수는 계산했지만 이 환경에서는 Grad-CAM을 표시하지 못했어요.';
      console.warn('Grad-CAM unavailable:', error.message);
    } finally { explainInput?.dispose(); }
    signal?.throwIfAborted();
    return { status: 'result', real: item.real, fake: item.fake, verdict: item.verdict, explanation, explanationError };
  } finally {
    signal?.removeEventListener('abort', cancel);
    cancel();
    URL.revokeObjectURL(url);
    img.removeAttribute('src');
    $('process-original').removeAttribute('src');
    const canvas = $('process-input'); canvas.width = canvas.height = 1;
  }
};
refreshHistoryViews();
if (routeName() === 'history') renderHistoryPage();
loadResources();
