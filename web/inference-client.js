// One worker / one model per tab. Serial RPC avoids overlapping tensor workloads.
let worker, serial = 0;
const pending = new Map();
export function releaseInference() {
  worker?.terminate();
  worker = null;
  for (const request of pending.values()) {
    request.reject(new DOMException('분석을 취소했어요.', 'AbortError'));
  }
  pending.clear();
  document.dispatchEvent(new Event('fakit:inference-released'));
}
export function metric(name, data = {}) { document.dispatchEvent(new CustomEvent('fakit:metric', {detail:{name,data}})); }
export function inference(type, payload = {}) {
  if (!worker) {
    const url=new URL('./inference-worker.js?v=classifiers', import.meta.url);
    if(new URLSearchParams(location.search).has('diagnostics'))url.searchParams.set('diagnostics','1');
    worker = new Worker(url);
    worker.onmessage = ({data}) => {
      if (data.metric) { metric(data.metric, data.data); return; }
      const request = pending.get(data.id); if (!request) return;
      pending.delete(data.id);
      data.error ? request.reject(new Error(data.error)) : request.resolve(data.value);
    };
    worker.onerror = () => {
      for (const request of pending.values()) request.reject(new Error('분석 실행기를 불러오지 못했어요. 다시 시도해 주세요.'));
      pending.clear(); worker.terminate(); worker = null;
    };
  }
  return new Promise((resolve,reject) => {
    const id = ++serial; pending.set(id,{resolve,reject}); worker.postMessage({id,type,payload});
  });
}
