// Browser adaptation of feature/generalization-study/scripts/11_gradcam_gravex_fn.py.
// The TF.js export folds top_conv + batch normalization. Explain its final spatial
// activation instead; do not claim pixel-level localization or a calibrated score.
const ROOT = 'StatefulPartitionedCall/functional_7_1/';
const FEATURE = ROOT + 'efficientnetv2-s_1/top_activation_1/mul_1';
export function fakeLogit(tf, features, weights) {
  const pooled = tf.concat([features.mean([1, 2]), features.max([1, 2])], 1);
  const dense = pooled.matMul(weights[0]).add(weights[1]);
  const hidden = dense.mul(weights[2]).add(weights[3]).relu();
  return hidden.matMul(weights[4]).add(weights[5]).neg().sum();
}
export async function computeGradcam(tf, model, input, expectedReal, { signal } = {}) {
  signal?.throwIfAborted();
  const names = ['dense_256_1/Cast/ReadVariableOp', 'dense_256_1/BiasAdd/ReadVariableOp',
    'bn_1/batchnorm/mul', 'bn_1/batchnorm/sub', 'probability_real_1/Cast/ReadVariableOp', 'probability_real_1/Add/ReadVariableOp'];
  // Clone graph constants: disposing these tensors must never dispose model weights.
  let features, weights = [], heat;
  try {
    features = model.execute(input, FEATURE);
    weights = names.map(name => model.execute(input, ROOT + name).clone());
    const actual = tf.tidy(() => fakeLogit(tf, features, weights).neg().sigmoid());
    let real;
    try { real = (await actual.data())[0]; } finally { actual.dispose(); }
    if (!Number.isFinite(real) || Math.abs(real - expectedReal) > 0.001) throw new Error('설명 모델과 판별 모델의 출력이 일치하지 않아요.');
    signal?.throwIfAborted();
    heat = tf.tidy(() => {
      const gradients = tf.grad(x => fakeLogit(tf, x, weights))(features);
      const pooled = gradients.mean([0, 1, 2]);
      const raw = features.squeeze([0]).mul(pooled).sum(-1).relu();
      return raw.div(raw.max().add(1e-8));
    });
    const values = Array.from(await heat.data());
    if (!values.every(Number.isFinite)) throw new Error('설명 지도를 계산하지 못했어요.');
    signal?.throwIfAborted();
    return { width: heat.shape[1], height: heat.shape[0], values, target: 'fake', layer: 'top_activation' };
  } finally { tf.dispose([features, ...weights, heat].filter(Boolean)); }
}

export function overlayGradcam(image, map) {
  const canvas = document.createElement('canvas'); canvas.width = canvas.height = 256;
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('설명 이미지를 그릴 수 없어요.');
  ctx.drawImage(image, 0, 0, 256, 256);
  const heat = document.createElement('canvas'); heat.width = map.width; heat.height = map.height;
  const hc = heat.getContext('2d'), pixels = hc.createImageData(map.width, map.height);
  map.values.forEach((v, i) => { pixels.data.set([255, Math.round(180 * (1-v)), 30, Math.round(v * 155)], i*4); });
  hc.putImageData(pixels, 0, 0); ctx.imageSmoothingEnabled = true; ctx.drawImage(heat, 0, 0, 256, 256);
  return canvas.toDataURL('image/png');
}
