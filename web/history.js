import { openGradcam, renderPhotoEvidence } from './gradcam-dialog.js?v=photo-result';

const $ = id => document.getElementById(id);
const MAX_RECORDS = 200;
const MAX_BYTES = 64 * 1024 * 1024;
const encoder = new TextEncoder();
const decoder = new TextDecoder();
let database;
let activeScope;
let revision = 0;
let filter = 'all';
let entries = [];

const request = operation => new Promise((resolve, reject) => {
  operation.onsuccess = () => resolve(operation.result);
  operation.onerror = () => reject(operation.error);
});

function openDatabase() {
  if (!database) {
    const opening = indexedDB.open('fakit-inspections', 1);
    opening.onupgradeneeded = () => {
      const db = opening.result;
      db.createObjectStore('records', { keyPath: ['scope', 'id'] }).createIndex('scope', 'scope');
      db.createObjectStore('results');
    };
    database = request(opening).catch(error => { database = null; throw error; });
  }
  return database;
}

export const captureHistoryOwner = () => window.Fakit.getHistoryAccess();
const isCurrent = owner => owner && owner.scope === captureHistoryOwner()?.scope;
const settingKey = owner => 'fakit.inspections.enabled.' + owner.scope;
const enabled = owner => {
  try { return !!owner && localStorage.getItem(settingKey(owner)) !== '0'; }
  catch { return false; }
};

async function transform(owner, id, part, value, writing) {
  if (!owner.key) return writing ? JSON.stringify(value) : JSON.parse(value);
  const bytes = Uint8Array.from(owner.key.match(/../g), byte => parseInt(byte, 16));
  const key = await crypto.subtle.importKey('raw', bytes, 'AES-GCM', false, [writing ? 'encrypt' : 'decrypt']);
  const additionalData = encoder.encode(JSON.stringify([owner.scope, id, part]));
  if (writing) {
    const iv = crypto.getRandomValues(new Uint8Array(12));
    const data = await crypto.subtle.encrypt({name:'AES-GCM', iv, additionalData}, key, encoder.encode(JSON.stringify(value)));
    return { iv, data };
  }
  const plain = await crypto.subtle.decrypt({name:'AES-GCM', iv:value.iv, additionalData}, key, value.data);
  return JSON.parse(decoder.decode(plain));
}

async function write(owner, update) {
  const db = await openDatabase();
  if (!isCurrent(owner)) return;
  const tx = db.transaction(['records', 'results'], 'readwrite');
  const done = new Promise((resolve, reject) => {
    tx.oncomplete = resolve;
    tx.onabort = () => reject(tx.error || new Error('저장이 취소됐어요.'));
  });
  done.catch(() => {});
  try {
    await update(tx.objectStore('records'), tx.objectStore('results'));
    await done;
  } catch (error) {
    try { tx.abort(); } catch {}
    await done.catch(() => {});
    throw error;
  }
}

export async function saveCompleted(owner, input, result, type) {
  if (result.status !== 'result' || !input.id || !isCurrent(owner) || !enabled(owner)) return;
  const { id } = input;
  const at = Date.now();
  const summary = {
    id, type, at, name: input.name || (type === 'image' ? '링크 사진' : '영상'),
    status:'completed', verdict:result.verdict, real:result.real, fake:result.fake,
    threshold:result.threshold ?? result.video?.threshold, model:result.model ?? result.video?.model,
    thumb: result.thumb ?? result.video?.thumbnail,
  };
  try {
    const meta = await transform(owner, id, 'meta', summary, true);
    const payload = await transform(owner, id, 'result', result, true);
    const size = [meta, payload].reduce((sum, value) => sum + (typeof value === 'string' ? value.length * 2 : value.data.byteLength + 12), 0);
    if (size > MAX_BYTES) throw new Error('이 검사 결과는 저장 한도를 넘어요.');
    await write(owner, async (records, results) => {
      const rows = await request(records.index('scope').getAll(owner.scope));
      if (!isCurrent(owner) || rows.some(row => row.id === id)) return;
      records.add({scope:owner.scope, id, at, size, meta});
      results.add(payload, [owner.scope, id]);
      let bytes = size + rows.reduce((sum,row) => sum + row.size, 0);
      rows.sort((a,b) => b.at - a.at);
      while (rows.length >= MAX_RECORDS || bytes > MAX_BYTES) {
        const old = rows.pop();
        records.delete([owner.scope, old.id]);
        results.delete([owner.scope, old.id]);
        bytes -= old.size;
      }
    });
    void refresh();
  } catch (error) {
    if (isCurrent(owner)) window.Fakit.toast(error.name === 'QuotaExceededError' ? '저장 공간이 부족해 이번 검사는 기록에 남기지 못했어요.' : '검사는 완료했지만 기록을 저장하지 못했어요.');
  }
}

async function erase(owner, id) {
  if (!owner) return;
  try {
    await write(owner, async (records, results) => {
      const keys = id ? [[owner.scope,id]] : await request(records.index('scope').getAllKeys(owner.scope));
      if (!isCurrent(owner)) return;
      for (const key of keys) { records.delete(key); results.delete(key); }
    });
    void refresh();
  } catch { window.Fakit.toast('기록을 삭제하지 못했어요.'); }
}

const typeName = entry => entry.type === 'video' ? '영상' : '사진';
const verdict = entry => entry.verdict === 'real' ? 'REAL' : 'AI 생성';
const date = at => new Date(at).toLocaleString('ko-KR');

async function openResult(entry) {
  const owner = captureHistoryOwner();
  if (!owner) return;
  try {
    const db = await openDatabase();
    const payload = await request(db.transaction('results').objectStore('results').get([owner.scope,entry.id]));
    if (!isCurrent(owner) || !payload) return;
    const result = await transform(owner, entry.id, 'result', payload, false);
    if (!isCurrent(owner)) return;
    $('saved-kind').textContent = typeName(entry) + ' 검사 결과';
    $('saved-title').textContent = entry.verdict === 'real' ? '실제 촬영으로 분류' : 'AI 생성으로 분류';
    $('saved-detail').dataset.verdict = entry.verdict;
    $('saved-meta').textContent = entry.name + ' · ' + date(entry.at);
    $('saved-ai').textContent = (result.fake * 100).toFixed(1) + '%';
    $('saved-real').textContent = (result.real * 100).toFixed(1) + '%';
    const threshold = result.video?.threshold ?? result.threshold;
    $('saved-score').textContent = (Number.isFinite(threshold)
      ? `판정 기준 · ${result.video ? 'AI' : '실제'} 점수 ${(threshold * 100).toFixed(0)}% · `
      : '') + '모델 점수는 진위가 확정될 확률이 아닙니다.';
    const video = $('saved-video');
    video.replaceChildren();
    video.hidden = !result.video;
    if (result.video) {
      const title = document.createElement('h3');
      title.textContent = '구간별 AI 점수';
      const summary = document.createElement('p');
      summary.textContent = result.video.duration_sec.toFixed(1) + '초 · ' + result.video.analyzed_frames + '프레임 검사';
      const segments = document.createElement('ul');
      for (const segment of [...result.video.segments, ...result.video.skipped_segments].sort((a,b)=>a.start_sec-b.start_sec)) {
        const line = document.createElement('li');
        const time = document.createElement('span');
        time.textContent = segment.start_sec.toFixed(1) + '–' + segment.end_sec.toFixed(1) + '초';
        const bar = document.createElement('span');
        bar.className = 'saved-segment-bar';
        bar.setAttribute('aria-hidden', 'true');
        const score = document.createElement('strong');
        score.textContent = '미분석';
        if (Number.isFinite(segment.fake_score)) {
          const percent = segment.fake_score * 100;
          bar.style.setProperty('--segment-score', `${Math.max(0, Math.min(100, percent))}%`);
          score.textContent = percent.toFixed(1) + '%';
        }
        line.append(time, bar, score);
        segments.append(line);
      }
      video.append(title, summary, segments);
    }
    const photo = $('saved-photo-evidence');
    photo.hidden = entry.type !== 'image';
    if (!photo.hidden) {
      renderPhotoEvidence($('saved-photo-images'), result);
      $('saved-photo-note').textContent = result.explanation
        ? '색이 진한 곳은 모델의 판정에 더 크게 기여한 영역이에요. 실제 조작 위치를 뜻하지는 않아요.'
        : '이 검사에는 저장된 Grad-CAM 결과가 없어요.';
    }
    $('saved-gradcam').hidden = !photo.hidden;
    $('saved-gradcam').onclick = () => openGradcam(result);
    window.Fakit.openModal($('saved-detail'));
  } catch { if (isCurrent(owner)) window.Fakit.toast('이 검사 결과를 불러오지 못했어요.'); }
}

$('saved-detail').addEventListener('close', () => $('saved-photo-images').replaceChildren());

function card(entry, removable) {
  const container = document.createElement('article');
  container.className = 'history-card';
  const button = document.createElement('button');
  button.type = 'button';
  button.className = 'recent-item';
  const tag = document.createElement('b');
  tag.textContent = typeName(entry) + ' · ' + verdict(entry);
  const name = document.createElement('span');
  name.className = 'recent-name';
  name.textContent = entry.name;
  const time = document.createElement('small');
  time.textContent = date(entry.at);
  if (entry.thumb) {
    const image = document.createElement('img');
    image.src = entry.thumb;
    image.alt = '';
    image.loading = 'lazy';
    button.append(image);
  }
  button.append(tag,name,time);
  button.onclick = () => void openResult(entry);
  container.append(button);
  if (removable) {
    const remove = document.createElement('button');
    remove.type = 'button';
    remove.className = 'history-remove';
    remove.setAttribute('aria-label', entry.name + ' 기록 삭제');
    remove.textContent = '×';
    remove.onclick = () => void erase(captureHistoryOwner(), entry.id);
    container.append(remove);
  }
  return container;
}

function render() {
  const route = window.Fakit.routeName();
  $('recent-strip').hidden = !entries.length || !['home','image','video'].includes(route);
  $('recent-list').replaceChildren(...entries.slice(0,8).map(entry => card(entry,false)));
  const shown = entries.filter(entry => filter === 'all' || entry.verdict === filter);
  $('history-list').replaceChildren(...shown.map(entry => card(entry,true)));
  $('history-empty').hidden = shown.length > 0;
  $('history-empty').querySelector('strong').textContent = entries.length ? '이 판정에 해당하는 기록이 없습니다.' : '아직 기록이 없습니다.';
  $('history-count').textContent = entries.length ? entries.length + '건 저장됨' : '';
  $('history-clear').hidden = !entries.length;
  $('dock-history-count').hidden = !entries.length;
  $('dock-history-count').textContent = entries.length > 99 ? '99+' : entries.length;
  const owner = captureHistoryOwner();
  $('history-toggle').disabled = !owner;
  $('history-toggle').checked = enabled(owner);
}

async function refresh() {
  const owner = captureHistoryOwner();
  const token = ++revision;
  if (activeScope !== owner?.scope) {
    activeScope = owner?.scope;
    entries = [];
    $('gradcam-dialog').close();
    $('saved-detail').close();
    render();
  }
  if (!owner) { render(); return; }
  try {
    const db = await openDatabase();
    const rows = await request(db.transaction('records').objectStore('records').index('scope').getAll(owner.scope));
    const decoded = await Promise.allSettled(rows.map(row => transform(owner,row.id,'meta',row.meta,false)));
    if (token !== revision || !isCurrent(owner)) return;
    entries = decoded.filter(item => item.status === 'fulfilled').map(item => item.value).sort((a,b) => b.at-a.at);
    render();
  } catch { if (token === revision) render(); }
}

$('saved-detail').addEventListener('close', () => {
  $('saved-gradcam').onclick = null;
  $('saved-video').replaceChildren();
});
$('history-toggle').onchange = () => {
  const owner = captureHistoryOwner();
  if (!owner) return;
  try { localStorage.setItem(settingKey(owner), $('history-toggle').checked ? '1' : '0'); }
  catch { window.Fakit.toast('기록 저장 설정을 바꾸지 못했어요.'); render(); }
};
$('history-clear').onclick = () => {
  if (confirm('현재 기록을 모두 삭제할까요?')) void erase(captureHistoryOwner());
};
document.querySelectorAll('.chip[data-filter]').forEach(button => button.onclick = () => {
  filter = button.dataset.filter;
  document.querySelectorAll('.chip[data-filter]').forEach(item => item.classList.toggle('is-on', item === button));
  render();
});
document.addEventListener('fakit:user', () => void refresh());
document.addEventListener('fakit:route', render);
addEventListener('pageshow', event => { if (event.persisted) void refresh(); });
void refresh();
