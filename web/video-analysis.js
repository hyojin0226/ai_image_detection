import { openGradcam } from './gradcam-dialog.js';
const $ = id => document.getElementById(id);

export async function analyzeVideo(file, { signal, onState = () => {} } = {}) {
  onState('loading', '영상 분석 모델을 불러오고 있어요.');
  const { analyzeFile } = await import('./browser-engine.js?v=accuracy');
  signal?.throwIfAborted();
  const video = await analyzeFile(file, { signal, progress: message => {
    onState(message.includes('준비') ? 'loading' : 'analyzing', message);
    updateVideoProgress(message);
  } });
  signal?.throwIfAborted();
  if (!Number.isFinite(video.fake_score) || video.fake_score < 0 || video.fake_score > 1) throw Error('영상 모델의 결과를 읽지 못했어요.');
  return { status: 'result', real: 1 - video.fake_score, fake: video.fake_score, verdict: video.prediction, video };
}

export function updateVideoProgress(message = '') {
  const progress = $('video-progress');
  const match = message.match(/(?:프레임 추출|Grad-CAM)\s+(\d+)\/(\d+)/);
  progress.hidden = !match;
  if (match) { progress.max = Number(match[2]); progress.value = Number(match[1]); progress.setAttribute('aria-label', message); }
}

function frameCard(frame, result) {
  const figure = document.createElement('figure');
  const button = document.createElement('button'); button.type = 'button';
  button.setAttribute('aria-label', `${frame.timestamp_sec.toFixed(2)}초 원본과 Grad-CAM 확대`);
  const image = document.createElement('img'); image.loading = 'lazy'; image.src = frame.comparison_url;
  image.alt = `${frame.timestamp_sec.toFixed(2)}초 · 원본과 ${frame.target_class === 'fake' ? 'AI 생성' : '실제 촬영'} 판정 기여도`;
  button.append(image);
  button.onclick = () => openGradcam(result,frame.index);
  const caption = document.createElement('figcaption');
  caption.textContent = `${frame.timestamp_sec.toFixed(2)}초 · 구간 ${frame.segment}${frame.gradcam_available ? '' : ' · 양의 기여 신호 없음'}`;
  figure.append(button, caption); return figure;
}

export function renderVideoResult(result) {
  const panel = $('video-result'), video = result?.video;
  panel.hidden = !video;
  $('video-main-frames').replaceChildren(); $('video-detail-frames').replaceChildren(); $('video-segments').replaceChildren();
  $('video-details').open = false; $('video-details').ontoggle = null;
  updateVideoProgress();
  if (!video) return;
  $('video-meta').textContent = `${video.duration_sec.toFixed(1)}초 · ${video.analyzed_frames}/${video.planned_frames}프레임 분석 · ${video.elapsed_sec}초 소요`;
  $('video-limits').textContent = video.limitations_ko;
  const map = new Map(video.frames.map(frame => [frame.index, frame]));
  $('video-main-frames').append(...video.main_frame_indices.map(id => map.get(id)).filter(Boolean).map(frame => frameCard(frame,result)));
  $('video-details-title').textContent = `나머지 프레임 ${video.detail_frame_indices.length}개`;
  let rendered = false;
  $('video-details').ontoggle = () => {
    if (!$('video-details').open || rendered) return;
    rendered = true; $('video-detail-frames').append(...video.detail_frame_indices.map(id => map.get(id)).filter(Boolean).map(frame => frameCard(frame,result)));
  };
  const segments = [...video.segments, ...video.skipped_segments].sort((a,b) => a.start_sec - b.start_sec);
  for (const segment of segments) {
    const node = document.createElement('div');
    const score = segment.fake_score;
    node.className = 'video-segment'; node.dataset.verdict = score == null ? 'unknown' : score >= .5 ? 'fake' : 'real';
    node.textContent = `${segment.start_sec.toFixed(1)}–${segment.end_sec.toFixed(1)}초 · ${score == null ? '미분석' : 'AI 점수 ' + (score * 100).toFixed(1) + '%'}`;
    $('video-segments').append(node);
  }
}
