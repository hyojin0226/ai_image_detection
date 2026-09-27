// Browser port of streamlit_app.py: same model, same preprocessing, same threshold.
const MODEL_URL = "model/model.json";
const CONFIG_URL = "model/inference_config.json";

const $ = (id) => document.getElementById(id);
const els = {
  status: $("model-status"), statusText: $("status-text"), loadFill: $("loadbar-fill"),
  dropzone: $("dropzone"), input: $("file-input"), empty: $("dz-empty"), preview: $("preview"),
  error: $("error"), result: $("result"), analyzing: $("analyzing"), body: $("result-body"),
  prediction: $("prediction"), fakeVal: $("fake-val"), realVal: $("real-val"),
  fakeBar: $("fake-bar"), realBar: $("real-bar"), thresholdMark: $("threshold-mark"),
  thresholdText: $("threshold-text"), reset: $("reset"),
};

let model = null;
let config = null;
let pendingFile = null;

async function loadResources() {
  try {
    config = await (await fetch(CONFIG_URL)).json();
    model = await tf.loadGraphModel(MODEL_URL, {
      onProgress: (p) => { els.loadFill.style.width = `${Math.round(p * 100)}%`; },
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
    els.statusText.textContent =
      "모델을 불러오지 못했습니다. 로컬에서 열었다면 index.html을 직접 열지 말고 웹 서버로 실행하세요 (README 참고).";
  }
}

function showError(msg) {
  els.error.textContent = msg;
  els.error.hidden = false;
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
    els.empty.hidden = true;
    els.preview.hidden = false;
    els.dropzone.classList.add("has-image");
    els.result.hidden = false;
    els.analyzing.hidden = false;
    els.body.hidden = true;
    if (model) analyze(els.preview);
    else {
      pendingFile = els.preview;
      els.analyzing.lastChild.textContent = " 모델을 불러오는 중… 끝나면 바로 분석합니다.";
    }
  };
  els.preview.onerror = () => {
    URL.revokeObjectURL(url);
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

async function analyze(img) {
  pendingFile = null;
  els.analyzing.lastChild.textContent = " 분석 중…";
  await new Promise(requestAnimationFrame);  // let the spinner paint

  const input = toInputTensor(img);
  const output = model.predict(input);
  const probabilityReal = (await output.data())[0];
  tf.dispose([input, output]);

  const threshold = Number(config.decision_threshold_real);
  const probabilityFake = 1 - probabilityReal;
  const isReal = probabilityReal >= threshold;

  els.prediction.textContent = isReal ? "REAL" : "AI GENERATED / FAKE";
  els.prediction.className = `prediction ${isReal ? "is-real" : "is-fake"}`;
  els.fakeVal.textContent = `${(probabilityFake * 100).toFixed(2)}%`;
  els.realVal.textContent = `${(probabilityReal * 100).toFixed(2)}%`;
  els.fakeBar.style.width = "0";
  els.realBar.style.width = "0";
  els.thresholdMark.style.left = `${threshold * 100}%`;
  els.thresholdText.textContent =
    `REAL 판정 기준선(threshold): ${threshold.toFixed(3)} — REAL 확률이 이 값 이상이면 REAL로 판정합니다.`;

  els.analyzing.hidden = true;
  els.body.hidden = false;
  requestAnimationFrame(() => {
    els.fakeBar.style.width = `${probabilityFake * 100}%`;
    els.realBar.style.width = `${probabilityReal * 100}%`;
  });
}

function reset() {
  els.input.value = "";
  els.preview.hidden = true;
  els.preview.removeAttribute("src");
  els.empty.hidden = false;
  els.dropzone.classList.remove("has-image");
  els.result.hidden = true;
  els.error.hidden = true;
  pendingFile = null;
}

els.input.addEventListener("change", (e) => handleFile(e.target.files[0]));
els.dropzone.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") { e.preventDefault(); els.input.click(); }
});
["dragenter", "dragover"].forEach((t) => els.dropzone.addEventListener(t, (e) => {
  e.preventDefault();
  els.dropzone.classList.add("drag");
}));
["dragleave", "drop"].forEach((t) => els.dropzone.addEventListener(t, (e) => {
  e.preventDefault();
  els.dropzone.classList.remove("drag");
}));
els.dropzone.addEventListener("drop", (e) => handleFile(e.dataTransfer.files[0]));
els.reset.addEventListener("click", reset);

loadResources();
