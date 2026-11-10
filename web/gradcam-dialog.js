const $ = id => document.getElementById(id);
const dialog = $('gradcam-dialog');
const framePicker = $('gradcam-frame');
const gallery = $('gradcam-images');
const missing = '이 검사에는 저장된 Grad-CAM 결과가 없습니다';
let selection = 0;

export function scoreText(result) {
  const scores = 'REAL ' + (result.real * 100).toFixed(1) + '% · AI ' + (result.fake * 100).toFixed(1) + '%';
  const threshold = result.video ? '영상 AI 기준 ' + (result.video.threshold * 100).toFixed(0) + '%' :
    Number.isFinite(result.threshold) ? '사진 REAL 기준 ' + (result.threshold * 100).toFixed(0) + '%' : '';
  return [scores, threshold, '점수는 모델 출력(0–1)을 백분율로 표시한 값입니다.'].filter(Boolean).join(' · ');
}

function imagePanel(label, src) {
  const figure = document.createElement('figure');
  const img = document.createElement('img');
  img.src = src;
  img.alt = label;
  const caption = document.createElement('figcaption');
  caption.textContent = label;
  figure.append(img,caption);
  return figure;
}

function showImages(original, heatmap, overlay) {
  gallery.replaceChildren(imagePanel('원본 미리보기', original), imagePanel('Grad-CAM 히트맵', heatmap), imagePanel('겹쳐 보기', overlay));
}

function canvas(width,height) {
  const node = document.createElement('canvas');
  node.width = width;
  node.height = height;
  return node;
}

async function showFrame(frame, result) {
  const token = ++selection;
  gallery.replaceChildren();
  $('gradcam-score').textContent = scoreText(result) +
    ' · 프레임 ' + frame.timestamp_sec.toFixed(2) + '초 · 프레임 이미지 AI 점수 ' +
    (frame.image_fake_score * 100).toFixed(1) + '%';
  if (!frame.gradcam_available || !frame.heatmap) {
    $('gradcam-message').textContent = frame.note || missing;
    if (frame.comparison_url) gallery.append(imagePanel('저장된 프레임 비교', frame.comparison_url));
    return;
  }
  $('gradcam-message').textContent = '표시 대상 · ' + (frame.target_class === 'fake' ? 'AI 생성' : 'REAL');
  try {
    const comparison = new Image();
    comparison.src = frame.comparison_url;
    await comparison.decode();
    if (token !== selection || !dialog.open) return;
    const crop = x => {
      const target = canvas(256,256);
      target.getContext('2d').drawImage(comparison,x,24,256,256,0,0,256,256);
      return target.toDataURL();
    };
    const heat = canvas(8,8);
    const context = heat.getContext('2d');
    const pixels = context.createImageData(8,8);
    frame.heatmap.forEach((value,index) => pixels.data.set([255, Math.round(180*(1-value)), 30, Math.round(value*255)], index*4));
    context.putImageData(pixels,0,0);
    showImages(crop(0),heat.toDataURL(),crop(256));
  } catch {
    if (token === selection) $('gradcam-message').textContent = '저장된 프레임 이미지를 읽지 못했어요.';
  }
}

export function openGradcam(result, frameIndex) {
  ++selection;
  framePicker.replaceChildren();
  gallery.replaceChildren();
  $('gradcam-message').textContent = '';
  $('gradcam-score').textContent = result ? scoreText(result) : '';
  const frames = result?.video?.frames || [];
  $('gradcam-frame-label').hidden = !frames.length;
  window.Fakit.openModal(dialog);
  if (frames.length) {
    frames.forEach(frame => framePicker.add(new Option(frame.timestamp_sec.toFixed(2) + '초 · 구간 ' + frame.segment, frame.index)));
    const first = frames.find(frame => frame.index === frameIndex) || frames.find(frame => frame.gradcam_available) || frames[0];
    framePicker.value = first.index;
    framePicker.onchange = () => void showFrame(frames.find(frame => String(frame.index) === framePicker.value), result);
    void showFrame(first,result);
  } else if (result?.explanation?.original && result.explanation.heatmap && result.explanation.image) {
    $('gradcam-message').textContent = '표시 대상 · ' + (result.explanation.target === 'real' ? 'REAL' : 'AI 생성');
    showImages(result.explanation.original,result.explanation.heatmap,result.explanation.image);
  } else {
    $('gradcam-message').textContent = missing;
  }
}

document.querySelectorAll('[data-close-dialog]').forEach(button => button.onclick = () => button.closest('dialog').close());
dialog.addEventListener('close', () => {
  ++selection;
  framePicker.onchange = null;
  framePicker.replaceChildren();
  gallery.replaceChildren();
});
