import { fetchImageFile } from './image-import.js';
import { youtubeVideoId } from './media-links.js';

export const LIMITS = Object.freeze({ imageBytes: 15 * 1024 * 1024, textChars: 20000 });
export const VIDEO_SERVICE = 'https://duhyeon-ai-video.vercel.app/';
const videoHandoff = () => ({ status: 'unsupported', serviceUrl: VIDEO_SERVICE, message: '영상은 팀 영상 분석기에서 검사할 수 있어요. 아래 버튼으로 이동해 파일이나 유튜브 Shorts 주소를 입력해 주세요.' });
export function classifyInput(payload, source = 'typed') {
  if (typeof payload === 'string') {
    const value = payload.trim();
    let kind = 'text';
    if (value.length <= 16384 && /^https?:\/\/\S+$/i.test(value)) kind = 'url';
    return { kind, value, source };
  }
  const kind = payload.type?.startsWith('image/') ? 'image' : payload.type?.startsWith('video/') ? 'video' : 'unknown';
  return { kind, file: payload, source };
}

/** Result: {status:'result', real, fake, verdict} | {status:'unsupported', message}.
 * Missing models deliberately return no scores, sentences, frames or progress. */
export async function analyze(input, { signal, onState = () => {}, onResolved = () => {}, imageAnalyzer = globalThis.Fakit?.analyzeImage, fetchImpl } = {}) {
  signal?.throwIfAborted();
  if (input.kind === 'text') {
    return { status: 'unsupported', message: '글은 분석하지 않아요. 얼굴 사진이나 영상 링크를 넣어 주세요.' };
  }
  if (input.kind === 'video') return videoHandoff();
  if (input.kind === 'url') {
    if (youtubeVideoId(input.value)) return videoHandoff();
    onState('loading', '링크에서 파일을 확인하고 있어요.');
    try {
      const file = await fetchImageFile(input.value, { signal, fetchImpl });
      signal?.throwIfAborted();
      onResolved(file);
      return analyze({ kind: 'image', file, source: 'url' }, { signal, onState, imageAnalyzer });
    } catch (error) {
      if (error.code === 'VIDEO') return videoHandoff();
      if (error.code === 'PAGE') return { status: 'unsupported', message: '웹페이지 분석은 아직 지원하지 않아요. 이미지 파일의 직접 주소를 넣어 주세요.' };
      throw error;
    }
  }
  if (input.kind !== 'image') return { status: 'unsupported', message: '이 파일 형식은 아직 지원하지 않아요. 이미지나 영상을 선택해 주세요.' };
  if (!input.file?.type.startsWith('image/')) throw new Error('이미지 파일을 선택해 주세요. 종류를 바꿔도 파일 형식은 변환되지 않아요.');
  if (input.file.size > LIMITS.imageBytes) throw new Error('이미지는 15MB 이하만 검사할 수 있어요.');
  if (input.source !== 'paste' && !['image/jpeg', 'image/png', 'image/webp'].includes(input.file.type)) throw new Error('현재 이미지 검사는 JPG·PNG·WEBP를 지원해요.');
  if (!imageAnalyzer) throw new Error('분석 도구를 불러오지 못했어요. 새로고침 후 다시 시도해 주세요.');
  return imageAnalyzer(input.file, { signal, onState });
}
