// Browser port of streamlit_app.py: same model, same preprocessing, same threshold.
const MODEL_URL = "model/model.json";
const CONFIG_URL = "model/inference_config.json";
// Inference alone takes a fraction of a second; keep the scan on screen long enough to read.
const MIN_SCAN_MS = 1200;

const $ = (id) => document.getElementById(id);
const els = {
  status: $("model-status"), statusText: $("status-text"), loadFill: $("loadbar-fill"),
  modelError: $("model-error"),
  card: $("card"), dropzone: $("dropzone"), input: $("file-input"), preview: $("preview"),
  shotBlur: $("shot-blur"), error: $("error"), analyzingText: $("analyzing-text"),
  prediction: $("prediction"), verdictSub: $("verdict-sub"), fakeVal: $("fake-val"), realVal: $("real-val"),
  fakeBar: $("fake-bar"), realBar: $("real-bar"), thresholdMark: $("threshold-mark"),
  thresholdText: $("threshold-text"), reset: $("reset"),
};
const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;

let model = null;
let config = null;
let pendingFile = null;
let previewUrl = null;
let runId = 0;   // bumped on every new image / reset so a stale analysis can't overwrite a newer one

async function loadResources() {
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
    if (pendingFile) analyze(pendingFile);
  } catch (err) {
    console.error(err);
    els.status.classList.add("failed");
    els.statusText.textContent = "모델을 불러오지 못했습니다";
    els.modelError.textContent = "네트워크 연결을 확인한 뒤 새로고침해 주세요.";
    els.modelError.hidden = false;
  }
}

function showError(msg) {
  els.error.textContent = msg;
  els.error.hidden = false;
}

function setVerdict(verdict) {
  if (verdict) {
    els.card.dataset.verdict = verdict;
    document.body.dataset.verdict = verdict;   // tints the backdrop glow
  } else {
    delete els.card.dataset.verdict;
    delete document.body.dataset.verdict;
  }
}

function setPreviewUrl(url) {
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = url;
}

function handleFile(file) {
  els.error.hidden = true;
  if (!file) return;
  if (!/^image\/(jpeg|png|webp)$/.test(file.type)) {
    showError("이미지 파일을 읽을 수 없습니다. JPG, PNG 또는 WEBP 파일을 사용하세요.");
    return;
  }
  const url = URL.createObjectURL(file);
  els.preview.onload = () => {
    setPreviewUrl(url);
    els.shotBlur.style.backgroundImage = `url("${url}")`;
    els.preview.hidden = false;
    setVerdict(null);
    els.card.dataset.state = "scanning";
    els.card.scrollIntoView({ behavior: "smooth", block: "nearest" });
    if (model) analyze(els.preview);
    else {
      pendingFile = els.preview;
      els.analyzingText.textContent = "모델을 불러오는 중… 끝나면 바로 분석합니다.";
    }
  };
  els.preview.onerror = () => {
    URL.revokeObjectURL(url);
    if (els.preview.getAttribute("src") !== url) return;   // src was cleared by reset()
    reset();
    showError("이미지 파일을 읽을 수 없습니다. JPG, PNG 또는 WEBP 파일을 사용하세요.");
  };
  els.preview.src = url;
}

// Equivalent of PIL image.convert("RGB").resize(image_size) → float32 array (0–255, no normalization;
// the model's own Rescaling layer handles that).
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

// requestAnimationFrame stalls while the tab is in the background; never let that hold up a result.
function nextFrame() {
  return new Promise((resolve) => {
    const fallback = setTimeout(resolve, 100);
    requestAnimationFrame(() => { clearTimeout(fallback); resolve(); });
  });
}

function countUp(el, percent, run) {
  const final = `${percent.toFixed(2)}%`;
  if (reduceMotion || document.hidden) { el.textContent = final; return; }
  const start = performance.now();
  const duration = 1100;
  const tick = (now) => {
    if (run !== runId) return;
    const t = Math.min((now - start) / duration, 1);
    el.textContent = t < 1 ? `${(percent * (1 - Math.pow(1 - t, 4))).toFixed(2)}%` : final;
    if (t < 1) requestAnimationFrame(tick);
  };
  el.textContent = "0.00%";
  requestAnimationFrame(tick);
  setTimeout(() => { if (run === runId) el.textContent = final; }, duration + 200);
}

async function analyze(img) {
  pendingFile = null;
  const run = ++runId;
  const started = performance.now();
  els.analyzingText.textContent = "분석 중…";
  await nextFrame();  // let the scan paint

  const input = toInputTensor(img);
  const output = model.predict(input);
  const probabilityReal = (await output.data())[0];
  tf.dispose([input, output]);

  const remaining = reduceMotion ? 0 : MIN_SCAN_MS - (performance.now() - started);
  if (remaining > 0) await new Promise((resolve) => setTimeout(resolve, remaining));
  if (run !== runId) return;

  const threshold = Number(config.decision_threshold_real);
  const probabilityFake = 1 - probabilityReal;
  const isReal = probabilityReal >= threshold;

  els.prediction.textContent = isReal ? "REAL" : "AI GENERATED";
  els.prediction.className = `prediction ${isReal ? "is-real" : "is-fake"}`;
  els.verdictSub.textContent = isReal ? "실제 사진으로 판정했습니다." : "AI가 생성한 얼굴로 판정했습니다.";
  els.fakeBar.style.width = "0";
  els.realBar.style.width = "0";
  els.thresholdMark.style.left = `${threshold * 100}%`;
  els.thresholdText.textContent = `REAL 확률이 기준선 ${threshold.toFixed(3)} 이상이면 REAL로 판정합니다.`;

  setVerdict(isReal ? "real" : "fake");
  els.card.dataset.state = "done";
  countUp(els.fakeVal, probabilityFake * 100, run);
  countUp(els.realVal, probabilityReal * 100, run);
  await nextFrame();  // widths were reset to 0 above; change them a frame later so they animate
  if (run !== runId) return;
  // 1px shy of each share so the two fills meet with a hairline gap.
  els.fakeBar.style.width = `calc(${probabilityFake * 100}% - 1px)`;
  els.realBar.style.width = `calc(${probabilityReal * 100}% - 1px)`;
}

function reset() {
  runId++;
  pendingFile = null;
  els.input.value = "";
  els.preview.hidden = true;
  els.preview.removeAttribute("src");
  els.shotBlur.style.backgroundImage = "";
  setPreviewUrl(null);
  setVerdict(null);
  els.card.dataset.state = "idle";
  els.error.hidden = true;
}

els.input.addEventListener("change", (e) => handleFile(e.target.files[0]));
els.dropzone.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") { e.preventDefault(); els.input.click(); }
});
els.reset.addEventListener("click", reset);

// Files can be dropped anywhere on the page, or pasted.
const hasFiles = (e) => e.dataTransfer && [...e.dataTransfer.types].includes("Files");
["dragenter", "dragover"].forEach((t) => document.addEventListener(t, (e) => {
  if (!hasFiles(e)) return;
  e.preventDefault();
  els.card.classList.add("drag");
}));
document.addEventListener("dragleave", (e) => {
  if (!e.relatedTarget) els.card.classList.remove("drag");   // left the window
});
document.addEventListener("drop", (e) => {
  if (!hasFiles(e)) return;
  e.preventDefault();
  els.card.classList.remove("drag");
  handleFile(e.dataTransfer.files[0]);
});
document.addEventListener("paste", (e) => {
  const file = [...(e.clipboardData?.files ?? [])].find((f) => f.type.startsWith("image/"));
  if (file) handleFile(file);
});

loadResources();
