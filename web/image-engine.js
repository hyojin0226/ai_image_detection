// Image page engine: BlazeFace face detection -> 2x face crop -> 256x256 -> ConvNeXt-T (ONNX).
// Same pipeline and runtime versions as the verified develop-branch module (web_model/test.html).
// The video page keeps the EfficientNetV2S TF.js worker because the video heads use its features.
import { MODEL_BASE } from './model-release.js';
import { createFaceModel } from './face-model.js?v=photo-stages';

const ORT_VERSION = '1.20.1';
const MEDIAPIPE_VERSION = '0.10.14';
const ORT_BASE = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
const MEDIAPIPE_BASE = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MEDIAPIPE_VERSION}/`;

let loading = null;

function loadScript(src) {
  return new Promise((resolve, reject) => {
    const script = document.createElement('script');
    script.src = src; script.async = true; script.crossOrigin = 'anonymous';
    script.onload = resolve;
    script.onerror = () => reject(new Error('분석 실행기를 불러오지 못했어요. 연결을 확인해 주세요.'));
    document.head.append(script);
  });
}

// Fetch the model with progress so the page can show the 56 MB download.
async function fetchModel(url, onProgress) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(new Error('모델 다운로드 시간이 초과됐어요. 연결을 확인하고 다시 검사해 주세요.')), 120000);
  try {
    return await readModel(url, onProgress, controller.signal);
  } finally {
    clearTimeout(timeout);
  }
}

async function readModel(url, onProgress, signal) {
  const response = await fetch(url, { signal });
  if (!response.ok) throw new Error('모델 파일을 불러오지 못했어요.');
  const total = Number(response.headers.get('content-length')) || 0;
  if (!response.body || !total) return new Uint8Array(await response.arrayBuffer());
  const reader = response.body.getReader(), bytes = new Uint8Array(total);
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    bytes.set(value, received); received += value.length;
    onProgress(Math.min(received / total, 1));
  }
  if (received !== total) throw new Error('모델 다운로드가 끝나지 않았어요. 다시 검사해 주세요.');
  return bytes;
}

export function loadImageEngine({ onProgress = () => {} } = {}) {
  if (loading) return loading;
  loading = (async () => {
    if (!globalThis.ort) await loadScript(ORT_BASE + 'ort.min.js');
    const ort = globalThis.ort;
    ort.env.wasm.wasmPaths = ORT_BASE;
    ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
    // Run ONNX in a worker so the page stays responsive during inference.
    ort.env.wasm.proxy = true;
    const vision = await import(/* webpackIgnore: true */ MEDIAPIPE_BASE + 'vision_bundle.mjs');
    const modelBytes = await fetchModel(MODEL_BASE + 'convnext_tiny_fake_face.onnx', onProgress);
    const model = await createFaceModel({
      ort,
      vision: { FilesetResolver: vision.FilesetResolver, FaceDetector: vision.FaceDetector },
      modelUrl: modelBytes,
      configUrl: MODEL_BASE + 'face_model_config.json',
      detectorModelUrl: MODEL_BASE + 'blaze_face_short_range.tflite',
      mediapipeWasmUrl: MEDIAPIPE_BASE + 'wasm',
    });
    return model;
  })().catch(error => { loading = null; throw error; });
  return loading;
}
