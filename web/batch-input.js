import { analyze } from './input-adapters.js';

export const MAX_ATTACHMENTS = 20;

/** Serial inference keeps the existing single-image model and its memory budget intact. */
export async function analyzeBatch(inputs, { signal, onItem = () => {}, analyzeImpl = analyze, ...options } = {}) {
  const results = [];
  for (let index = 0; index < inputs.length; index++) {
    signal?.throwIfAborted();
    onItem(index, 'analyzing');
    let result;
    try { result = await analyzeImpl(inputs[index], { ...options, signal }); }
    catch (error) {
      signal?.throwIfAborted();
      result = { status: 'error', message: error.message || '이 파일을 읽지 못했어요.' };
    }
    signal?.throwIfAborted();
    results.push(result);
    onItem(index, result.status, result);
  }
  return results;
}
