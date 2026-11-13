// Fakit 최종 얼굴 모델 (ConvNeXt-T, ONNX) 브라우저 추론 모듈.
// 원본: develop 저장소 web_model/fakit-face-model.js (서버와 판정 불일치 0/131 얼굴로 검증).
// 사이트용 추가: explain 결과에 투명 히트맵(gradcam_heatmap), ELA 생략 옵션(ela:false).
//
// server/app.py predict_image + face.py + model.py 와 같은 순서를 브라우저에서 재현한다.
//   긴 변 2048 제한 -> MediaPipe BlazeFace 검출(+ 큰 이미지 타일 재검출) -> 얼굴 x2.0 정사각 crop
//   -> 256x256 (Pillow BILINEAR 와 같은 계산) -> JPEG 품질 90 재인코딩 -> /255, ImageNet 정규화 -> ONNX
// 출력 형식도 서버 응답(JSON)과 같은 키를 쓴다.
//
// 필요: 전역 `ort` (onnxruntime-web), MediaPipe tasks-vision 의 FilesetResolver / FaceDetector.
//
//   import { createFaceModel } from './fakit-face-model.js';
//   const model = await createFaceModel({ ort, vision: { FilesetResolver, FaceDetector }, ... });
//   const result = await model.analyze(fileOrBlobOrBitmap, { mode: 'face', explain: true });

const MAX_IMAGE_SIDE = 2048;
const DETECT = {
  minConfidence: 0.5,
  cropScale: 2.0,
  cropShiftUp: 0.05,
  minFaceRatio: 0.04,
  maxFaces: 8,
  tileMinConfidence: 0.75,
  verifyMinConfidence: 0.6,
};

// ---------------------------------------------------------------------------
// 픽셀 버퍼: { width, height, data: Uint8ClampedArray(RGBA) }

function makeCanvas(width, height) {
  if (typeof OffscreenCanvas !== 'undefined') return new OffscreenCanvas(width, height);
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  return canvas;
}

function bitmapToPixels(bitmap) {
  const canvas = makeCanvas(bitmap.width, bitmap.height);
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(bitmap, 0, 0);
  const { data } = ctx.getImageData(0, 0, bitmap.width, bitmap.height);
  return { width: bitmap.width, height: bitmap.height, data };
}

function pixelsToCanvas(pixels) {
  const canvas = makeCanvas(pixels.width, pixels.height);
  canvas.getContext('2d').putImageData(new ImageData(pixels.data, pixels.width, pixels.height), 0, 0);
  return canvas;
}

function cropPixels(pixels, left, top, width, height) {
  const out = new Uint8ClampedArray(width * height * 4);
  for (let y = 0; y < height; y++) {
    const src = ((top + y) * pixels.width + left) * 4;
    out.set(pixels.data.subarray(src, src + width * 4), y * width * 4);
  }
  return { width, height, data: out };
}

// ---------------------------------------------------------------------------
// Pillow Image.resize(..., BILINEAR) 재현 (libImaging/Resample.c, 8bit 고정소수점)

const PRECISION_BITS = 32 - 8 - 2;
const PRECISION_SCALE = 2 ** PRECISION_BITS;
const HALF = 2 ** (PRECISION_BITS - 1);

function precomputeCoeffs(inSize, outSize) {
  const scale = inSize / outSize;
  const filterScale = Math.max(scale, 1);
  const support = 1.0 * filterScale;
  const bounds = new Int32Array(outSize * 2);
  const coeffs = [];
  for (let xx = 0; xx < outSize; xx++) {
    const center = (xx + 0.5) * scale;
    let xmin = Math.trunc(center - support + 0.5);
    if (xmin < 0) xmin = 0;
    let xmax = Math.trunc(center + support + 0.5);
    if (xmax > inSize) xmax = inSize;
    xmax -= xmin;
    const k = new Float64Array(xmax);
    let ww = 0;
    for (let x = 0; x < xmax; x++) {
      const t = Math.abs((x + xmin - center + 0.5) / filterScale);
      const w = t < 1 ? 1 - t : 0;
      k[x] = w;
      ww += w;
    }
    const kk = new Float64Array(xmax);
    for (let x = 0; x < xmax; x++) {
      const v = ww !== 0 ? k[x] / ww : k[x];
      kk[x] = v < 0 ? Math.trunc(-0.5 + v * PRECISION_SCALE) : Math.trunc(0.5 + v * PRECISION_SCALE);
    }
    bounds[xx * 2] = xmin;
    bounds[xx * 2 + 1] = xmax;
    coeffs.push(kk);
  }
  return { bounds, coeffs };
}

function clip8(value) {
  const v = Math.floor(value / PRECISION_SCALE);
  return v < 0 ? 0 : v > 255 ? 255 : v;
}

export function resizeBilinear(pixels, outWidth, outHeight) {
  if (pixels.width === outWidth && pixels.height === outHeight) return pixels;
  let current = pixels;
  if (outWidth !== current.width) {
    const { bounds, coeffs } = precomputeCoeffs(current.width, outWidth);
    const out = new Uint8ClampedArray(outWidth * current.height * 4);
    for (let y = 0; y < current.height; y++) {
      const row = y * current.width * 4;
      for (let x = 0; x < outWidth; x++) {
        const xmin = bounds[x * 2], count = bounds[x * 2 + 1], k = coeffs[x];
        let r = HALF, g = HALF, b = HALF;
        for (let i = 0; i < count; i++) {
          const p = row + (xmin + i) * 4;
          r += current.data[p] * k[i];
          g += current.data[p + 1] * k[i];
          b += current.data[p + 2] * k[i];
        }
        const o = (y * outWidth + x) * 4;
        out[o] = clip8(r); out[o + 1] = clip8(g); out[o + 2] = clip8(b); out[o + 3] = 255;
      }
    }
    current = { width: outWidth, height: current.height, data: out };
  }
  if (outHeight !== current.height) {
    const { bounds, coeffs } = precomputeCoeffs(current.height, outHeight);
    const out = new Uint8ClampedArray(current.width * outHeight * 4);
    for (let y = 0; y < outHeight; y++) {
      const ymin = bounds[y * 2], count = bounds[y * 2 + 1], k = coeffs[y];
      for (let x = 0; x < current.width; x++) {
        let r = HALF, g = HALF, b = HALF;
        for (let i = 0; i < count; i++) {
          const p = ((ymin + i) * current.width + x) * 4;
          r += current.data[p] * k[i];
          g += current.data[p + 1] * k[i];
          b += current.data[p + 2] * k[i];
        }
        const o = (y * current.width + x) * 4;
        out[o] = clip8(r); out[o + 1] = clip8(g); out[o + 2] = clip8(b); out[o + 3] = 255;
      }
    }
    current = { width: current.width, height: outHeight, data: out };
  }
  return current;
}

// ---------------------------------------------------------------------------
// JPEG 재인코딩 (서버 model._jpeg: PIL quality=90)

async function encodeJpeg(canvas, quality) {
  if (canvas.convertToBlob) return canvas.convertToBlob({ type: 'image/jpeg', quality });
  return new Promise((resolve, reject) =>
    canvas.toBlob((blob) => (blob ? resolve(blob) : reject(new Error('JPEG 인코딩 실패'))), 'image/jpeg', quality));
}

async function decodeBlob(blob) {
  return createImageBitmap(blob, { imageOrientation: 'from-image', colorSpaceConversion: 'none', premultiplyAlpha: 'none' });
}

async function jpegRoundTrip(pixels, quality) {
  if (!quality) return pixels;
  const blob = await encodeJpeg(pixelsToCanvas(pixels), quality / 100);
  const bitmap = await decodeBlob(blob);
  try {
    return bitmapToPixels(bitmap);
  } finally {
    bitmap.close?.();
  }
}

// ---------------------------------------------------------------------------
// 얼굴 검출 (server/face.py FaceDetector.detect 와 같은 규칙)

function centerInside(a, b) {
  const cx = a.x + a.w / 2, cy = a.y + a.h / 2;
  return b.x <= cx && cx <= b.x + b.w && b.y <= cy && cy <= b.y + b.h;
}

function iou(a, b) {
  const x1 = Math.max(a.x, b.x), y1 = Math.max(a.y, b.y);
  const x2 = Math.min(a.x + a.w, b.x + b.w), y2 = Math.min(a.y + a.h, b.y + b.h);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const union = a.w * a.h + b.w * b.h - inter;
  return union > 0 ? inter / union : 0;
}

function pyRange(start, stop, step) {
  const out = [];
  for (let v = start; v < stop; v += step) out.push(v);
  return out;
}

class FaceFinder {
  constructor(detector) {
    this.detector = detector;
  }

  region(pixels, left = 0, top = 0, width = pixels.width, height = pixels.height) {
    const part = left || top || width !== pixels.width || height !== pixels.height
      ? cropPixels(pixels, left, top, width, height) : pixels;
    const result = this.detector.detect(new ImageData(part.data, part.width, part.height));
    return result.detections.map((d) => ({
      x: d.boundingBox.originX + left,
      y: d.boundingBox.originY + top,
      w: d.boundingBox.width,
      h: d.boundingBox.height,
      score: d.categories?.[0]?.score ?? 0,
    }));
  }

  verify(pixels, box) {
    const { width, height } = pixels;
    const side = Math.trunc(Math.max(box.w, box.h) * 2.5);
    const cx = box.x + box.w / 2, cy = box.y + box.h / 2;
    const left = Math.trunc(Math.min(Math.max(cx - side / 2, 0), Math.max(0, width - side)));
    const top = Math.trunc(Math.min(Math.max(cy - side / 2, 0), Math.max(0, height - side)));
    // numpy 슬라이싱처럼 이미지 밖은 잘라낸다
    const w = Math.min(side, width - left), h = Math.min(side, height - top);
    return this.region(pixels, left, top, w, h)
      .some((found) => found.score >= DETECT.verifyMinConfidence && centerInside(found, box));
  }

  async detect(pixels, signal) {
    const { width, height } = pixels;
    let boxes = this.region(pixels);
    if (Math.min(width, height) >= 400) {
      const tile = Math.trunc(Math.min(width, height) * 0.35);
      const step = Math.max(1, Math.trunc(tile / 2));
      const tops = [...new Set([...pyRange(0, Math.max(1, height - tile + 1), step), Math.max(0, height - tile)])].sort((a, b) => a - b);
      const lefts = [...new Set([...pyRange(0, Math.max(1, width - tile + 1), step), Math.max(0, width - tile)])].sort((a, b) => a - b);
      let lastYield = performance.now();
      for (const top of tops) {
        for (const left of lefts) {
          if (performance.now() - lastYield >= 24) {
            await new Promise(resolve => setTimeout(resolve, 0));
            signal?.throwIfAborted();
            lastYield = performance.now();
          }
          const w = Math.min(tile, width - left), h = Math.min(tile, height - top);
          for (const box of this.region(pixels, left, top, w, h)) {
            if (box.score >= DETECT.tileMinConfidence && this.verify(pixels, box)) boxes.push(box);
          }
        }
      }
    }
    const minSide = DETECT.minFaceRatio * Math.min(width, height);
    boxes = boxes.filter((b) => Math.max(b.w, b.h) >= minSide);
    boxes.sort((a, b) => b.score - a.score); // 안정 정렬: 파이썬 sort 와 같은 순서
    const kept = [];
    for (const box of boxes) {
      if (kept.every((k) => iou(box, k) < 0.3 && !centerInside(box, k) && !centerInside(k, box))) kept.push(box);
    }
    return kept.slice(0, DETECT.maxFaces);
  }
}

// 파이썬 round() (짝수 반올림)
function pyRound(value) {
  const floor = Math.floor(value);
  const diff = value - floor;
  if (diff > 0.5) return floor + 1;
  if (diff < 0.5) return floor;
  return floor % 2 === 0 ? floor : floor + 1;
}

export function cropBoxFor(face, width, height) {
  let side = Math.max(face.w, face.h) * DETECT.cropScale;
  side = Math.min(side, width, height);
  const cx = face.x + face.w / 2;
  const cy = face.y + face.h / 2 - DETECT.cropShiftUp * side;
  const left = Math.min(Math.max(cx - side / 2, 0), width - side);
  const top = Math.min(Math.max(cy - side / 2, 0), height - side);
  return [pyRound(left), pyRound(top), pyRound(side), pyRound(side)];
}

// ---------------------------------------------------------------------------
// Grad-CAM 후처리 (server model.gradcam + app.gradcam_overlay)

const CAM_ANCHORS = [[0, 0, 0.5], [0, 0.3, 1], [0, 0.9, 0.9], [0.5, 1, 0.3], [1, 0.85, 0], [1, 0.3, 0], [0.6, 0, 0]];
const CAM_LUT = Array.from({ length: 256 }, (_, i) => {
  const t = (i / 255) * (CAM_ANCHORS.length - 1);
  const j = Math.min(Math.floor(t), CAM_ANCHORS.length - 2);
  const f = t - j;
  return CAM_ANCHORS[j].map((v, c) => (v + (CAM_ANCHORS[j + 1][c] - v) * f) * 255);
});

function camToHeatmap(camMap, side, target, size) {
  const n = side * side;
  const values = new Float32Array(n);
  let max = 0;
  for (let i = 0; i < n; i++) {
    const v = Math.max(target === 'fake' ? -camMap[i] : camMap[i], 0);
    values[i] = v;
    if (v > max) max = v;
  }
  if (max <= 0) return new Float32Array(size * size);
  const small = makeCanvas(side, side);
  const img = new ImageData(side, side);
  for (let i = 0; i < n; i++) {
    const v = Math.trunc((values[i] / max) * 255);
    img.data[i * 4] = img.data[i * 4 + 1] = img.data[i * 4 + 2] = v;
    img.data[i * 4 + 3] = 255;
  }
  small.getContext('2d').putImageData(img, 0, 0);
  const big = makeCanvas(size, size);
  const ctx = big.getContext('2d', { willReadFrequently: true });
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(small, 0, 0, size, size);
  const data = ctx.getImageData(0, 0, size, size).data;
  const out = new Float32Array(size * size);
  for (let i = 0; i < size * size; i++) out[i] = data[i * 4] / 255;
  return out;
}

// 히트맵만 따로 (색은 overlay 와 같고, 세기만큼 불투명)
function heatPixels(heat, size) {
  const data = new Uint8ClampedArray(size * size * 4);
  for (let i = 0; i < heat.length; i++) {
    const color = CAM_LUT[Math.trunc(heat[i] * 255)];
    for (let c = 0; c < 3; c++) data[i * 4 + c] = Math.round(color[c]);
    data[i * 4 + 3] = Math.round(255 * Math.min(Math.max(heat[i] * 1.5, 0), 1));
  }
  return { width: size, height: size, data };
}

function overlay(pixels, heat, alpha = 0.5) {
  const out = new Uint8ClampedArray(pixels.data.length);
  for (let i = 0; i < heat.length; i++) {
    const color = CAM_LUT[Math.trunc(heat[i] * 255)];
    const w = alpha * Math.min(Math.max(heat[i] * 1.5, 0), 1);
    for (let c = 0; c < 3; c++) out[i * 4 + c] = Math.round(pixels.data[i * 4 + c] * (1 - w) + color[c] * w);
    out[i * 4 + 3] = 255;
  }
  return { width: pixels.width, height: pixels.height, data: out };
}

// ---------------------------------------------------------------------------
// ELA (server/ela.py): JPEG 90 으로 다시 저장했을 때의 픽셀 오차. 모델 판정과 무관한 참고 지표

const ELA_QUALITY = 90;
const ELA_ANCHORS = [[0, 0, 4], [40, 11, 84], [101, 21, 110], [159, 42, 99], [212, 72, 66], [245, 125, 21], [250, 193, 39], [252, 255, 164]];
const ELA_LUT = Array.from({ length: 256 }, (_, i) => {
  const t = (i / 255) * (ELA_ANCHORS.length - 1);
  const j = Math.min(Math.floor(t), ELA_ANCHORS.length - 2);
  const f = t - j;
  return ELA_ANCHORS[j].map((v, c) => Math.trunc(v + (ELA_ANCHORS[j + 1][c] - v) * f));
});

/** 픽셀별 |원본 - 재저장| 의 세 채널 합 (0~765). 채널 평균 = 합 / 3 */
async function errorLevels(pixels) {
  const resaved = await jpegRoundTrip(pixels, ELA_QUALITY);
  const n = pixels.width * pixels.height;
  const sums = new Uint16Array(n);
  for (let i = 0; i < n; i++) {
    const p = i * 4;
    sums[i] = Math.abs(pixels.data[p] - resaved.data[p]) + Math.abs(pixels.data[p + 1] - resaved.data[p + 1])
      + Math.abs(pixels.data[p + 2] - resaved.data[p + 2]);
  }
  return { width: pixels.width, height: pixels.height, sums };
}

function percentileFromHistogram(hist, count, q) {
  // numpy.percentile(method='linear') 와 같은 값. 값은 정수 합 / 3
  const position = (q / 100) * (count - 1);
  const lower = Math.floor(position);
  const frac = position - lower;
  const valueAt = (rank) => {
    let seen = 0;
    for (let s = 0; s < hist.length; s++) {
      seen += hist[s];
      if (seen > rank) return s / 3;
    }
    return (hist.length - 1) / 3;
  };
  const a = valueAt(lower);
  return frac ? a + (valueAt(Math.min(lower + 1, count - 1)) - a) * frac : a;
}

function elaMap(levels, [left, top, width, height], size) {
  const hist = new Uint32Array(766);
  let regionSum = 0;
  for (let y = top; y < top + height; y++) {
    for (let x = left; x < left + width; x++) {
      const s = levels.sums[y * levels.width + x];
      hist[s]++;
      regionSum += s;
    }
  }
  const count = width * height;
  const p99 = percentileFromHistogram(hist, count, 99);
  const scale = Math.max(p99, 1);
  const colored = new Uint8ClampedArray(count * 4);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const v = levels.sums[(top + y) * levels.width + left + x] / 3;
      const color = ELA_LUT[Math.trunc(Math.min(Math.max(v / scale, 0), 1) * 255)];
      const o = (y * width + x) * 4;
      colored[o] = color[0]; colored[o + 1] = color[1]; colored[o + 2] = color[2]; colored[o + 3] = 255;
    }
  }
  let imageSum = 0;
  for (let i = 0; i < levels.sums.length; i++) imageSum += levels.sums[i];
  const image = resizeBilinear({ width, height, data: colored }, size, size);
  return {
    image,
    stats: { mean: round(regionSum / 3 / count, 3), p99: round(p99, 3), image_mean: round(imageSum / 3 / levels.sums.length, 3), quality: ELA_QUALITY },
  };
}

async function pixelsToDataUrl(pixels, type = 'image/jpeg') {
  const canvas = pixelsToCanvas(pixels);
  const blob = type === 'image/png'
    ? await (canvas.convertToBlob ? canvas.convertToBlob({ type }) : new Promise((resolve) => canvas.toBlob(resolve, type)))
    : await encodeJpeg(canvas, 0.92);
  return new Promise((resolve) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.readAsDataURL(blob);
  });
}

// ---------------------------------------------------------------------------

function round(value, digits = 4) {
  const f = 10 ** digits;
  return Math.round(value * f) / f;
}

// 단계별 검증(test.html)용
export const internals = { bitmapToPixels, decodeBlob, jpegRoundTrip, resizeBilinear, MAX_IMAGE_SIDE, pyRound };

export async function createFaceModel({
  ort,
  vision,
  modelUrl,
  configUrl,
  detectorModelUrl,
  mediapipeWasmUrl,
  executionProviders = ['wasm'],
}) {
  const config = await (await fetch(configUrl)).json();
  const size = config.input_size;
  const thresholds = config.decision_threshold_real;
  const fileset = await vision.FilesetResolver.forVisionTasks(mediapipeWasmUrl);
  const detector = await vision.FaceDetector.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: detectorModelUrl, delegate: 'CPU' },
    runningMode: 'IMAGE',
    minDetectionConfidence: DETECT.minConfidence,
  });
  let session;
  try {
    session = await ort.InferenceSession.create(modelUrl, { executionProviders, graphOptimizationLevel: 'all' });
  } catch (error) {
    detector.close();
    throw error;
  }
  const finder = new FaceFinder(detector);
  const [mean, std] = [config.mean, config.std];

  function toTensor(list) {
    const plane = size * size;
    const array = new Float32Array(list.length * 3 * plane);
    list.forEach((pixels, n) => {
      const base = n * 3 * plane;
      for (let i = 0; i < plane; i++) {
        for (let c = 0; c < 3; c++) {
          array[base + c * plane + i] = (pixels.data[i * 4 + c] / 255 - mean[c]) / std[c];
        }
      }
    });
    return new ort.Tensor('float32', array, [list.length, 3, size, size]);
  }

  async function runModel(inputs) {
    const input = toTensor(inputs);
    let outputs;
    try {
      outputs = await session.run({ [config.input_name]: input });
      const logits = outputs.logit.data;
      const cams = outputs.cam_map.data;
      const side = outputs.cam_map.dims[1];
      return inputs.map((_, i) => ({
        probReal: 1 / (1 + Math.exp(-logits[i])),
        cam: cams.slice(i * side * side, (i + 1) * side * side),
        camSide: side,
      }));
    } finally {
      input.dispose();
      for (const output of Object.values(outputs ?? {})) output.dispose();
    }
  }

  /** 서버의 /predict 응답과 같은 형식으로 돌려준다. source: File | Blob | ImageBitmap | {width,height,data} */
  async function analyze(source, { mode = 'face', explain = false, ela = true, signal, onStage = () => {} } = {}) {
    if (mode !== 'face' && mode !== 'full') throw new Error("mode 는 'face' 또는 'full' 입니다.");
    const started = performance.now();
    signal?.throwIfAborted();
    onStage('사진을 읽고 있어요.');
    let original;
    if (source && source.data && source.width) {
      original = source;
    } else {
      const bitmap = source instanceof ImageBitmap ? source : await decodeBlob(source);
      try {
        original = bitmapToPixels(bitmap);
      } finally {
        if (bitmap !== source) bitmap.close();
      }
    }
    const { width, height } = original;
    let scale = 1;
    let image = original;
    if (Math.max(width, height) > MAX_IMAGE_SIDE) {
      scale = MAX_IMAGE_SIDE / Math.max(width, height);
      image = resizeBilinear(original, pyRound(width * scale), pyRound(height * scale));
    }
    const workW = image.width, workH = image.height;

    let faces, cropBoxes;
    if (mode === 'full') {
      faces = [null];
      cropBoxes = [[0, 0, workW, workH]];
    } else {
      onStage('사진에서 얼굴을 찾고 있어요.');
      faces = await finder.detect(image, signal);
      cropBoxes = faces.map((f) => cropBoxFor(f, workW, workH));
    }
    const base = { image_size: [width, height], threshold_real: thresholds[mode], mode };
    if (!faces.length) {
      return { ...base, status: 'no_face', label: 'no_face', message: '판단 불가(얼굴 없음)', faces: [],
        elapsed_ms: round(performance.now() - started, 1) };
    }

    const crops = mode === 'full'
      ? [resizeBilinear(image, size, size)]
      : cropBoxes.map(([l, t, w, h]) => resizeBilinear(cropPixels(image, l, t, w, h), size, size));
    const inputs = [];
    for (const crop of crops) {
      signal?.throwIfAborted();
      inputs.push(await jpegRoundTrip(crop, config.eval_jpeg_quality));
    }
    signal?.throwIfAborted();
    onStage('모델이 얼굴을 분석하고 있어요.');
    const predictions = await runModel(inputs);
    signal?.throwIfAborted();
    onStage('판정과 판단 근거를 정리하고 있어요.');

    const toOriginal = ([x, y, w, h]) => [round(x / scale, 1), round(y / scale, 1), round(w / scale, 1), round(h / scale, 1)];
    const toNorm = ([x, y, w, h]) => [round(x / workW), round(y / workH), round(w / workW), round(h / workH)];

    const levels = explain && ela ? await errorLevels(image) : null;
    const results = [];
    for (let i = 0; i < faces.length; i++) {
      signal?.throwIfAborted();
      const face = faces[i];
      const { probReal, cam, camSide } = predictions[i];
      const bbox = face ? [face.x, face.y, face.w, face.h] : cropBoxes[i];
      const label = probReal >= thresholds[mode] ? 'real' : 'fake';
      const result = {
        bbox: toOriginal(bbox),
        bbox_norm: toNorm(bbox),
        crop_box: toOriginal(cropBoxes[i]),
        detection_score: face ? round(face.score) : null,
        prob_fake: round(1 - probReal),
        prob_real: round(probReal),
        prob_real_raw: probReal,
        label,
      };
      if (explain) {
        const heat = camToHeatmap(cam, camSide, label, size);
        result.crop_image = await pixelsToDataUrl(inputs[i]);
        result.gradcam_image = await pixelsToDataUrl(overlay(inputs[i], heat));
        result.gradcam_heatmap = await pixelsToDataUrl(heatPixels(heat, size), 'image/png');
        result.gradcam_target = label;
        if (levels) {
          const map = elaMap(levels, cropBoxes[i], size);
          result.ela_image = await pixelsToDataUrl(map.image);
          result.ela_stats = map.stats;
        }
      }
      results.push(result);
    }
    results.sort((a, b) => b.bbox[2] * b.bbox[3] - a.bbox[2] * a.bbox[3]);
    return { ...base, status: 'ok', label: results[0].label, faces: results, elapsed_ms: round(performance.now() - started, 1) };
  }

  return { analyze, config, close: () => detector.close() };
}
