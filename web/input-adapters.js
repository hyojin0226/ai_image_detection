import { fetchImageFile } from './image-import.js';

export const LIMITS = Object.freeze({ imageBytes: 15 * 1024 * 1024, textChars: 20000 });
export function classifyInput(payload, source = 'typed') {
  if (typeof payload === 'string') {
    const value = payload.trim();
    let kind = 'text';
    if (value.length < 200 && /^https?:\/\/\S+$/i.test(value)) kind = 'url';
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
    if (input.value?.length > LIMITS.textChars) throw new Error('글은 20,000자까지 입력할 수 있어요.');
    return { status: 'unsupported', message: '글 분석은 아직 지원하지 않아요. 모델을 준비하고 있어요.' };
  }
  if (input.kind === 'video') return { status: 'unsupported', message: '영상 분석은 아직 지원하지 않아요. 모델을 준비하고 있어요.' };
  if (input.kind === 'url') {
    onState('loading', '링크에서 파일을 확인하고 있어요.');
    try {
      const file = await fetchImageFile(input.value, { signal, fetchImpl });
      signal?.throwIfAborted();
      onResolved(file);
      return analyze({ kind: 'image', file, source: 'url' }, { signal, onState, imageAnalyzer });
    } catch (error) {
      if (error.code === 'VIDEO') return { status: 'unsupported', message: '영상 링크예요. 영상 분석은 아직 지원하지 않아요.' };
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
