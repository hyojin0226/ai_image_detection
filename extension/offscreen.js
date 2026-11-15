// 확장 안에서 모델을 돌리는 offscreen 문서. 서버 없이 얼굴 검출(MediaPipe) + 판정(EfficientNetV2S ONNX).
import { FilesetResolver, FaceDetector } from "./lib/mediapipe/vision_bundle.mjs";
import { createFaceModel } from "./fakit-face-model.js";

const url = (path) => chrome.runtime.getURL(path);

/* global ort */
ort.env.wasm.wasmPaths = url("lib/ort/");
ort.env.wasm.numThreads = 1;

let modelPromise = null;
function getModel() {
  if (!modelPromise) {
    modelPromise = createFaceModel({
      ort,
      vision: { FilesetResolver, FaceDetector },
      modelUrl: url("model/efficientnetv2s_fake_face.onnx"),
      configUrl: url("model/model_config.json"),
      detectorModelUrl: url("model/blaze_face_short_range.tflite"),
      mediapipeWasmUrl: url("lib/mediapipe"),
    }).catch((error) => {
      modelPromise = null; // 다음 요청에서 다시 시도
      throw error;
    });
  }
  return modelPromise;
}

async function dataUrlToBlob(dataUrl) {
  return (await fetch(dataUrl)).blob();
}

async function handle(message) {
  const model = await getModel();
  if (message.type === "health") {
    return { status: "ok", model: model.config.model_name, threshold_real: model.config.decision_threshold_real };
  }
  if (message.type === "analyze") {
    const blob = await dataUrlToBlob(message.dataUrl);
    return model.analyze(blob, { mode: message.mode || "face", explain: Boolean(message.explain) });
  }
  throw new Error(`알 수 없는 요청: ${message.type}`);
}

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message?.target !== "offscreen") return false;
  handle(message)
    .then((result) => sendResponse({ ok: true, result }))
    .catch((error) => sendResponse({ ok: false, error: String(error?.message || error) }));
  return true;
});
