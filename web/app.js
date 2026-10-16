// Detection: same model, preprocessing and threshold as streamlit_app.py, run in the browser with
// TF.js. Images are queued and analysed one at a time; results are kept in a local history.
const MODEL_URL = "model/model.json";
const CONFIG_URL = "model/inference_config.json";
const MAX_QUEUE = 20;                 // images per batch
const MAX_BYTES = 15 * 1024 * 1024;   // per image
const MIN_SCAN_MS = 650;              // inference is near-instant; keep the scan visible long enough to read
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
function toInputTensor(img) {
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
  return tf.tidy(() => tf.browser.fromPixels(canvas, 3).toFloat().expandDims(0));
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
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
  return pasted ? file.type.startsWith("image/") : /^image\/(jpeg|png|webp)$/.test(file.type);
}

async function addFiles(fileList, { pasted = false } = {}) {
  const files = [...fileList];
  if (!files.length) return;
  els.error.hidden = true;
  if (routeName() !== "home") go("home");

  const problems = [];
  const accepted = [];
  for (const file of files) {
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
    if (i === 0) els.queueSection.scrollIntoView({ behavior: "smooth", block: "nearest" });
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
      try {
        await modelReady;
      } catch {
        state.items.filter((i) => i.status !== "done").forEach((i) => { i.status = "error"; i.error = "모델 없음"; renderItem(i); });
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
  const started = performance.now();
  await nextFrame();   // let the scan paint
  try {
    const input = toInputTensor(item.img);
    const output = model.predict(input);
    const probabilityReal = (await output.data())[0];
    tf.dispose([input, output]);
    const remaining = reduceMotion ? 0 : MIN_SCAN_MS - (performance.now() - started);
    if (remaining > 0) await sleep(remaining);
    if (!state.items.includes(item)) return;   // removed while analysing
    item.real = probabilityReal;
    item.fake = 1 - probabilityReal;
    item.verdict = probabilityReal >= threshold() ? "real" : "fake";
    item.status = "done";
    renderItem(item);
    saveToHistory(item);
  } catch (err) {
    console.error(err);
    if (!state.items.includes(item)) return;
    item.status = "error";
    item.error = "분석 실패";
    renderItem(item);
  }
}

// ── Queue rendering ────────────────────────────────────────────────────
function cardHTML(entry, { removable = true, date = false } = {}) {
  return `
    ${removable ? `<button type="button" class="shot-remove" aria-label="목록에서 제거">×</button>` : ""}
    <div class="shot-img"><img src="${entry.url || entry.thumb}" alt=""><span class="scan" aria-hidden="true"></span><span class="shot-chip"></span></div>
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
    prob.textContent = pct(isReal ? entry.real : entry.fake);
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
function removeItem(item) {
  const idx = state.items.indexOf(item);
  if (idx < 0) return;
  state.items.splice(idx, 1);
  item.el.remove();
  URL.revokeObjectURL(item.url);
  renderSummary();
}
function clearQueue() {
  state.items.forEach((i) => { i.el.remove(); URL.revokeObjectURL(i.url); });
  state.items.length = 0;
  renderSummary();
}
function renderSummary() {
  const n = state.items.length;
  els.queueSection.hidden = n === 0;
  if (!n) return;
  const real = state.items.filter((i) => i.verdict === "real").length;
  const fake = state.items.filter((i) => i.verdict === "fake").length;
  const pending = state.items.filter((i) => i.status === "queued" || i.status === "analyzing").length;
  const failed = state.items.filter((i) => i.status === "error").length;
  els.summaryMain.textContent = pending ? `${n}장 중 ${n - pending}장 분석 완료` : `${n}장 분석 완료`;
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
  els.detailSub.textContent = isReal ? "실제 사진으로 판정했습니다." : "AI가 생성한 얼굴로 판정했습니다.";
  els.detailReal.textContent = pct(entry.real);
  els.detailFake.textContent = pct(entry.fake);
  els.detailThreshold.style.left = `${t * 100}%`;
  els.detailThresholdText.textContent = `REAL 확률이 기준선 ${t.toFixed(2)} 이상이면 REAL로 판정합니다.`;
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
    el.innerHTML = `<img src="${entry.thumb}" alt=""><b>${entry.verdict === "real" ? "REAL" : "AI"}</b>`;
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

// ── Inputs: file picker, drag & drop, paste ────────────────────────────
els.input.addEventListener("change", (e) => { addFiles(e.target.files); e.target.value = ""; });
els.inputMore.addEventListener("change", (e) => { addFiles(e.target.files); e.target.value = ""; });
els.addMore.addEventListener("click", () => els.inputMore.click());
els.clearQueue.addEventListener("click", clearQueue);
els.dropzone.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") { e.preventDefault(); els.input.click(); }
});

// Files can be dropped anywhere on the page (any route), or pasted.
const hasFiles = (e) => e.dataTransfer && [...e.dataTransfer.types].includes("Files");
["dragenter", "dragover"].forEach((t) => document.addEventListener(t, (e) => {
  if (!hasFiles(e)) return;
  e.preventDefault();
  if (routeName() !== "home") go("home");
  els.dropzone.classList.add("drag");
}));
document.addEventListener("dragleave", (e) => {
  if (!e.relatedTarget) els.dropzone.classList.remove("drag");   // left the window
});
document.addEventListener("drop", (e) => {
  if (!hasFiles(e)) return;
  e.preventDefault();
  els.dropzone.classList.remove("drag");
  addFiles(e.dataTransfer.files);
});

// Ctrl+V anywhere on the page: a screen capture (Win+Shift+S, PrtSc), an image copied from a
// browser or app, or image files copied in Explorer.
const NO_CLIPBOARD_IMAGE = "클립보드에 이미지가 없습니다. 화면을 캡처(Win+Shift+S)하거나 이미지를 복사한 뒤 다시 붙여넣어 주세요.";
const isEditable = (el) => el && (el.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(el.tagName));
document.addEventListener("paste", (e) => {
  if (isEditable(e.target)) return;
  const data = e.clipboardData;
  if (!data) return;
  // Some browsers only expose a capture through items, not files.
  let files = [...data.files].filter((f) => f.type.startsWith("image/"));
  if (!files.length) files = [...data.items].filter((i) => i.kind === "file" && i.type.startsWith("image/")).map((i) => i.getAsFile()).filter(Boolean);
  e.preventDefault();
  if (files.length) addFiles(files, { pasted: true });
  else { if (routeName() !== "home") go("home"); showError(NO_CLIPBOARD_IMAGE); }
});

// Button alternative for when the keyboard shortcut isn't handy (needs localhost/https).
els.pasteBtn.hidden = !navigator.clipboard?.read;
els.pasteBtn.addEventListener("click", async (e) => {
  e.preventDefault();   // the button sits inside the dropzone <label>; don't open the file picker
  e.stopPropagation();
  els.error.hidden = true;
  try {
    const files = [];
    for (const item of await navigator.clipboard.read()) {
      const type = item.types.find((t) => t.startsWith("image/"));
      if (!type) continue;
      files.push(new File([await item.getType(type)], "clipboard", { type }));
    }
    if (files.length) addFiles(files, { pasted: true });
    else showError(NO_CLIPBOARD_IMAGE);
  } catch (err) {
    console.error(err);
    showError(err.name === "NotAllowedError"
      ? "클립보드 접근이 거부되었습니다. 브라우저 권한을 허용하거나 Ctrl+V로 붙여넣어 주세요."
      : NO_CLIPBOARD_IMAGE);
  }
});

refreshHistoryViews();
if (routeName() === "history") renderHistoryPage();
loadResources();
