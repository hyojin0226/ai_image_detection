// Opt-in, local-only Performance/Resource Timing evidence. No files or pixels recorded.
if (new URLSearchParams(location.search).has('diagnostics')) {
  const events = [], longTasks = []; let workerRequests=[];
  const round = n => Math.round(n * 10) / 10;
  new PerformanceObserver(list => {
    for (const e of list.getEntries()) longTasks.push({ start: round(e.startTime), ms: round(e.duration) });
  }).observe({ type: 'longtask', buffered: true });
  document.addEventListener('fakit:metric', e => {
    if(e.detail.name==='worker-requests'){workerRequests=e.detail.data.requests;return;}
    if(e.detail.name==='model-progress')return;
    events.push({ name: e.detail.name, at: round(performance.now()), ...e.detail.data });
  });
  document.addEventListener('change', e => {
    if (e.target.type === 'file') events.push({ name: 'file-selected', at: round(performance.now()) });
  }, true);
  document.addEventListener('DOMContentLoaded', () => {
    const panel = document.createElement('details'); panel.id = 'performance-audit';
    const title = document.createElement('summary'); title.textContent = '성능 진단 (이 기기에만 표시)';
    const output = document.createElement('pre'); output.id = 'performance-data';
    const ping = document.createElement('button'); ping.textContent = '입력 응답 확인';
    ping.onclick = () => events.push({ name: 'interaction', at: round(performance.now()) });
    panel.append(title, ping, output); document.body.append(panel);
    panel.style.cssText = 'position:relative;z-index:100;padding:20px;background:white;color:black;overflow:auto;font:12px monospace';
    const draw = () => {
      output.textContent = JSON.stringify({
        paints: performance.getEntriesByType('paint').map(e => ({name:e.name,at:round(e.startTime)})),
        events, longTasks, workerRequests,
        requests: performance.getEntriesByType('resource').filter(e => /\/model\/|tf\.min\.js/.test(e.name)).map(e => ({url:e.name,start:round(e.startTime),ms:round(e.duration),bytes:e.transferSize,body:e.decodedBodySize})),
      }, null, 2);
    };
    setInterval(draw, 1000); draw();
  });
}
