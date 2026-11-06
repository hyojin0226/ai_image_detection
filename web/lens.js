import { analyze, classifyInput, LIMITS } from './input-adapters.js';
import { analyzeBatch, MAX_ATTACHMENTS } from './batch-input.js';
const $ = id => document.getElementById(id);
const stage = $('composer'), art = $('lens-art'), surface = $('lens-stage'), lens = $('lens-target');
const field = $('unified-input'), submit = $('inspect-button');
const reduced = matchMedia('(prefers-reduced-motion: reduce)');
const touch = matchMedia('(pointer: coarse)').matches;
const labels = { image: '이미지', video: '영상', text: '지원하지 않는 입력', url: '링크', unknown: '파일' };
let current = null, phase = 'idle', controller = null, generation = 0, previewUrl = null, video = null;
let width = 1000, height = 440, radius = 130, x = 500, y = 220, tx = 500, ty = 220;
let frame = 0, introStart = performance.now(), introDone = reduced.matches, dragging = 0, beforeDrag = 'idle';
let detectedType = null;
let attachments = [], textMode=false;
const inputForm=$('unified-form'), formHome=document.createComment('input form home');
inputForm.before(formHome);
stage.dataset.intro=reduced.matches?'false':'true';
function morphText(enabled) {
  if(textMode===enabled)return;
  textMode=enabled;
  const focused=document.activeElement===field, start=field.selectionStart, end=field.selectionEnd;
  stage.dataset.textMode=String(enabled);
  lens.tabIndex=enabled?-1:0; lens.setAttribute('aria-hidden',String(enabled));
  if(enabled) $('text-frame').append(inputForm); else formHome.after(inputForm);
  if(focused) {field.focus({preventScroll:true});field.setSelectionRange(start,end);}
  if(enabled) stage.scrollIntoView({block:'start',behavior:reduced.matches?'instant':'smooth'});
  layout();startMotion();
}
function clearAttachments() {
  attachments=[]; $('attachment-list').replaceChildren(); $('attachment-list').hidden=true;
  $('photo-deck').replaceChildren(); delete stage.dataset.multiple;
}
async function thumbnail(input, token) {
  if(!input.file?.type.startsWith('image/') || input.file.size>LIMITS.imageBytes)return;
  const url=URL.createObjectURL(input.file), img=new Image();
  try {
    img.src=url; await img.decode(); if(token!==generation)return;
    const canvas=document.createElement('canvas'); const scale=Math.min(1,160/Math.max(img.width,img.height));
    canvas.width=Math.max(1,Math.round(img.width*scale));canvas.height=Math.max(1,Math.round(img.height*scale));
    const ctx=canvas.getContext('2d');if(!ctx)return;
    ctx.drawImage(img,0,0,canvas.width,canvas.height); input.thumb=canvas.toDataURL('image/jpeg',.72);
    renderAttachments();
  } catch {} finally {URL.revokeObjectURL(url);img.removeAttribute('src');}
}
function renderExplanation(result) {
  const panel=$('explanation-panel'), image=$('explanation-image');
  panel.hidden=!result?.explanation && !result?.explanationError;
  image.hidden=!result?.explanation;
  if(result?.explanation) image.src=result.explanation.image; else image.removeAttribute('src');
  $('explanation-message').textContent=result?.explanationError || '같은 사진 모델의 실제 기울기로 계산했어요. 원본 사진은 서버에 보내지 않아요.';
}
function showResult(result) {
  renderExplanation(result);
  if(result.status!=='result') {delete stage.dataset.verdict;setState(result.status,result.message);return;}
  stage.dataset.verdict=result.verdict;
  $('lens-percent').textContent=(result.real*100).toFixed(1)+'%';
  $('lens-verdict').textContent=result.verdict==='real'?'실제 사진으로 분류':'AI 생성으로 분류';
  $('score-caption').textContent='REAL 점수';
  setState('result','점수는 모델 출력이며 진위의 확정이나 정답률이 아니에요.');
}
function renderAttachments() {
  const list=$('attachment-list'),deck=$('photo-deck');list.replaceChildren();deck.replaceChildren();
  list.hidden=attachments.length<2;stage.dataset.multiple=String(attachments.length>1);
  attachments.forEach((item,index)=>{
    const row=document.createElement('div');row.className='attachment';row.dataset.status=item.result?.status||item.status||'queued';
    const button=document.createElement('button');button.type='button';button.className='attachment-select';
    if(item.thumb){const img=document.createElement('img');img.src=item.thumb;img.alt='';button.append(img);}
    const name=document.createElement('span');name.textContent=item.name;button.append(name);
    const status=document.createElement('small');status.textContent=item.result?.status==='result'?(item.result.verdict==='real'?'실제 사진':'AI 생성')+' · REAL '+(item.result.real*100).toFixed(1)+'%':item.result?.message|| (item.status==='analyzing'?'검사 중':'대기');button.append(status);
    button.onclick=()=>{if(busy())return;if(item.result){current=item;detectedType=item.kind;showResult(item.result);}else {current=item;preview(item,generation);setState('idle');}};
    button.disabled=busy();row.append(button);
    if(!item.result){const remove=document.createElement('button');remove.type='button';remove.className='attachment-remove';remove.textContent='×';remove.setAttribute('aria-label',item.name+' 제거');remove.disabled=busy();remove.onclick=()=>{attachments.splice(index,1);if(!attachments.length){reset();return;}current=attachments[0];detectedType=current.kind;renderChip();renderAttachments();preview(current,generation);};row.append(remove);}
    list.append(row);
  });
  if(attachments.length>1) {
    const active=attachments.findIndex(item=>item.status==='analyzing');
    // A bounded fan gives the batch a physical shape without decoding full-size images.
    const deckStart=Math.max(0,active-2);
    attachments.slice(deckStart,deckStart+5).forEach((item,index)=>{
      const card=document.createElement('span');card.className='deck-card';card.style.setProperty('--card-index',index);card.style.setProperty('--card-offset',index-(Math.min(5,attachments.length)-1)/2);
      card.classList.toggle('is-current',index+deckStart===active);card.classList.toggle('is-done',!!item.result);
      if(item.thumb){const img=document.createElement('img');img.src=item.thumb;img.alt='';card.append(img);}else card.textContent=index+1;
      deck.append(card);
    });
  }
}

let vx=0, vy=0, lastTime=0, tiltX=0, tiltY=0;
function openInput(focus=false) {
  $('input-panel').hidden=false; stage.dataset.inputOpen='true';
  $('open-input').setAttribute('aria-expanded','true');
  if(focus) field.focus({preventScroll:true});
}
function closeInput() {
  $('input-panel').hidden=true; delete stage.dataset.inputOpen;
  $('open-input').setAttribute('aria-expanded','false');
}
$('open-input').onclick=()=>{openInput();morphText(true);field.focus({preventScroll:true});};
$('close-input').onclick=()=>{reset();lens.focus({preventScroll:true});};
const busy = () => ['loading', 'analyzing'].includes(phase);
function setState(next, message = '') {
  phase = next; stage.dataset.state = next;
  $('lens-message').textContent = message;
  $('cancel-analysis').hidden = !['loading', 'analyzing', 'detecting'].includes(next);
  $('retry-analysis').hidden = !['error', 'unsupported'].includes(next);
  $('reset-analysis').hidden = next !== 'result';
  $('lens-readout').hidden = !['result','error','unsupported'].includes(next);
  if (['error','unsupported'].includes(next)) { $('lens-percent').textContent=''; $('lens-verdict').textContent=message; $('score-caption').textContent=''; }
  submit.disabled = !current || current.kind === 'text' || busy() || next === 'result';
  lens.setAttribute('aria-label', busy() ? '검사 취소' : current ? '입력 검사' : '파일 선택');
  stage.setAttribute('aria-busy', String(busy()));
  $('kind-button').disabled=busy();
  layout(); startMotion();
}
function releasePreview() {
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = null;
  if (video) { video.pause(); video.removeAttribute('src'); video.load(); video = null; }
  $('lens-image').removeAttribute('href'); $('lens-image').setAttribute('visibility', 'hidden');
  $('lens-excerpt').setAttribute('visibility', 'hidden');
}
function abortRun() { generation++; controller?.abort(); controller = null; }
function reset() {
  renderExplanation(null); abortRun(); releasePreview(); clearAttachments(); morphText(false); current = null; detectedType = null;
  field.value = ''; field.style.height = ''; $('input-chip').hidden = true;
  $('input-note').textContent = ''; $('kind-menu').hidden = true; $('kind-button').setAttribute('aria-expanded','false');
  closeInput();
  delete stage.dataset.verdict; setState('idle');
}
function renderChip() {
  $('input-chip').hidden = !current;
  if (!current) return;
  const suffix = attachments.length>1 ? ' '+attachments.length+'개' : current.kind === 'text' ? ` ${(current.value || '').length.toLocaleString('ko-KR')}자` : current.kind === 'video' && current.duration ? ` ${Math.floor(current.duration / 60)}:${String(Math.floor(current.duration % 60)).padStart(2,'0')}` : '';
  $('kind-button').textContent = labels[current.kind] + suffix + ' ⌄';
  $('kind-button').setAttribute('aria-label', `${labels[current.kind]}${suffix}, 입력 종류 변경`);
}
async function preview(input, token) {
  releasePreview();
  if (input.file && ['image','video'].includes(input.kind)) {
    // Do not decode oversized images or videos when no video analyzer exists.
    if (input.kind === 'image' && input.file.size > LIMITS.imageBytes) return;
    previewUrl = URL.createObjectURL(input.file);
    if (input.kind === 'image') {
      $('lens-image').setAttribute('href', previewUrl);
      $('lens-image').setAttribute('visibility', 'visible');
    } else {
      const element = document.createElement('video'); video = element;
      element.preload = 'metadata'; element.muted = true; element.playsInline = true;
      element.onloadedmetadata = () => {
        if (token !== generation || current !== input) return;
        if (Number.isFinite(element.duration)) { input.duration = element.duration; renderChip(); }
        // No frame sampling or invented video progress without a video model.
      };
      element.src = previewUrl;
      $('lens-excerpt').textContent = '영상'; $('lens-excerpt').setAttribute('visibility', 'visible');
    }
  } else if (input.kind === 'text') {
    $('lens-excerpt').textContent = input.value.slice(0, 12);
    $('lens-excerpt').setAttribute('visibility','visible');
  }
  layout();
}
export async function handleInput(source, payload) {
  renderExplanation(null);
  const isFiles=typeof payload!=='string';
  const retain=isFiles && attachments.length && phase!=='result' && attachments.every(item=>!item.result);
  abortRun(); releasePreview();
  introDone=true; stage.dataset.intro='false';
  if(!retain)clearAttachments();
  if(window.Fakit.routeName()!=='home')window.Fakit.go('home');
  if(isFiles) {
    const files=payload instanceof Blob?[payload]:[...payload];
    if(!files.length)return;
    const remaining=MAX_ATTACHMENTS-attachments.length;
    attachments.push(...files.slice(0,remaining).map(file=>({...classifyInput(file,source),name:file.name||'붙여넣은 이미지'})));
    current=attachments[0];field.value='';morphText(false);
    $('input-note').textContent=files.length>remaining?'최대 20장까지 검사해요. 초과한 파일은 제외했어요.':attachments.length>1?attachments.length+'개 파일을 순서대로 검사해요':current.name;
  } else {
    if(!payload.trim()) {reset();openInput();morphText(true);field.focus({preventScroll:true});return;}
    current=classifyInput(payload,source);field.value=payload;
    $('input-note').textContent=current.kind==='text'?'사진이나 영상의 http 또는 https 링크를 입력해 주세요.':''; morphText(true);
  }
  detectedType=current.kind;openInput();delete stage.dataset.verdict;
  setState('detecting');renderChip();renderAttachments();
  const token=generation;preview(current,token);setState('idle');
  (async()=>{for(const item of attachments){if(token!==generation)break;if(!item.thumb)await thumbnail(item,token);}})();
  if(source==='drop')await run();
}
async function runBatch() {
  abortRun();const token=generation;controller=new AbortController();const signal=controller.signal;
  const inputs=[...attachments];let activeIndex=0;
  setState('loading','여러 사진을 순서대로 검사할게요.');
  try {
    const results=await analyzeBatch(inputs,{signal,onState:(state,message)=>{if(token===generation)setState(state,(activeIndex+1)+' / '+inputs.length+' · '+message);},onItem:(index,status,result)=>{
      if(token!==generation)return;
      activeIndex=index;inputs[index].status=status;
      if(result){inputs[index].result=result;if(result.status==='result')delete inputs[index].file;}
      else {current=inputs[index];preview(current,token);if(!current.thumb)thumbnail(current,token);}
      renderAttachments();
    }});
    if(token!==generation)return;
    renderExplanation(null);releasePreview();const successful=results.filter(r=>r.status==='result').length;
    delete stage.dataset.verdict;
    $('lens-percent').textContent=successful+' / '+results.length;
    $('lens-verdict').textContent='검사 완료';$('score-caption').textContent='아래에서 사진별 결과 확인';
    current={kind:'image',source:'picker'};
    setState('result',successful+'개 분석 완료'+(successful<results.length?' · 분석하지 못한 파일의 안내를 확인해 주세요.':'. 사진을 누르면 개별 점수를 볼 수 있어요.'));
    renderAttachments();
  }catch(error){if(token===generation)setState('error',error.message||'검사를 완료하지 못했어요.');}
}
async function run() {
  if (!current || busy() || phase === 'result') return;
  if(attachments.length>1 && !attachments.some(item=>item.result)) {await runBatch();return;}
  abortRun(); const token = generation;
  controller = new AbortController(); const signal = controller.signal;
  const input = current;
  try {
    // Manual corrections never coerce binary files into a different format.
    if (input.file && input.kind !== detectedType) throw new Error('선택한 종류와 파일 형식이 달라요. 해당 종류의 파일을 다시 넣어 주세요.');
    if (!input.file && ['image','video'].includes(input.kind)) throw new Error('이미지·영상은 파일이나 직접 링크로 넣어 주세요.');
    setState('loading', '입력을 확인하고 있어요.');
    const result = await analyze(input, { signal, onResolved: file => { if(token === generation) { morphText(false); current = classifyInput(file,'url'); detectedType='image'; renderChip(); preview(current,token); } }, onState: (state, message) => { if (token === generation) setState(state, message); } });
    if (token !== generation) return;
    releasePreview();
    if (result.status === 'unsupported') { morphText(false);setState('unsupported', result.message);return; }
    renderExplanation(result);
    stage.dataset.verdict = result.verdict;
    $('lens-percent').textContent = `${(result.real * 100).toFixed(1)}%`;
    $('lens-verdict').textContent = result.verdict === 'real' ? '실제 사진으로 분류' : 'AI 생성으로 분류';
    $('score-caption').textContent = 'REAL 점수';
    setState('result', '검사가 끝났어요. 점수는 모델 출력이며 진위의 확정이나 정답률이 아니에요.');
    // Results/history keep only scores and the existing small local thumbnail.
    if (current.file) { const item=attachments.find(item=>item===current);if(item){item.result=result;delete item.file;}current = { kind: current.kind, source: current.source };submit.disabled = true;renderAttachments(); }
  } catch (error) {
    if (token !== generation) return;
    releasePreview();
    const messages = { 'Image too large': '링크의 이미지가 15MB를 넘어요.', 'Unsupported image': 'JPG·PNG·WEBP 이미지의 직접 주소를 넣어 주세요.', 'Invalid image URL': '올바른 http 또는 https 주소를 넣어 주세요.' };
    setState('error', messages[error.message] || (input.kind === 'url' ? '링크를 가져오지 못했어요. 접근 제한을 확인하거나 파일을 직접 넣어 주세요.' : error.message || '입력을 읽지 못했어요. 다시 시도해 주세요.'));
  }
}
$('unified-form').addEventListener('submit', e => { e.preventDefault(); run(); });
field.addEventListener('input', () => { handleInput('typed',field.value); field.style.height='auto'; field.style.height=`${Math.min(field.scrollHeight,120)}px`; });
field.addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) { e.preventDefault(); run(); } });

$('clear-input').onclick = () => { reset(); openInput(true); };
$('cancel-analysis').onclick = reset;
$('reset-analysis').onclick = () => { reset(); lens.focus({preventScroll:true}); };
$('retry-analysis').onclick = () => { if (current) run(); else field.focus(); };
$('kind-button').onclick = () => { const open = $('kind-menu').hidden; $('kind-menu').hidden = !open; $('kind-button').setAttribute('aria-expanded',String(open)); };
$('kind-menu').addEventListener('click', e => {
  if (!e.target.dataset.kind || !current) return;
  abortRun(); current.kind = e.target.dataset.kind; renderChip(); releasePreview();
  $('kind-menu').hidden = true; $('kind-button').setAttribute('aria-expanded','false');
  morphText(!current.file);setState('idle'); $('kind-button').focus();
});
$('pick-button').onclick = () => {
  if (!touch) { $('unified-file').click(); return; }
  const open = $('pick-menu').hidden; $('pick-menu').hidden = !open; $('pick-button').setAttribute('aria-expanded', String(open));
};
for (const [button,input] of [['pick-file','unified-file'],['take-photo','camera-file'],['pick-album','album-file']]) {
  $(button).onclick = () => { $(input).click(); $('pick-menu').hidden = true; $('pick-button').setAttribute('aria-expanded','false'); };
}
for (const id of ['unified-file','camera-file','album-file']) $(id).onchange = e => { if(e.target.files.length) handleInput('picker',e.target.files); e.target.value=''; };
lens.onclick = () => {
  if(busy() || phase==='result') {reset();return;}
  if(current) {run();return;}
  openInput();
  if(touch) { $('pick-menu').hidden=false; $('pick-button').setAttribute('aria-expanded','true'); }
  else $('unified-file').click();
};
document.addEventListener('keydown', e => { if (e.key === 'Escape' && !document.querySelector('dialog[open]')) { $('pick-menu').hidden=true; reset(); } });
document.addEventListener('paste', e => {
  if (document.querySelector('dialog[open]') || (e.target !== field && e.target.closest('input,textarea,[contenteditable="true"]'))) return;
  const data = e.clipboardData; if (!data) return;
  const files = [...data.items].filter(i=>i.kind==='file').map(i=>i.getAsFile()).filter(Boolean);
  if (files.length) { e.preventDefault(); handleInput('paste',files); }
  else if (e.target !== field) { const text = data.getData('text/plain'); if (text) {e.preventDefault(); handleInput('paste',text);} }
});
const acceptsDrag = e => [...(e.dataTransfer?.types || [])].some(t=>['Files','text/plain','text/uri-list'].includes(t));
document.addEventListener('dragenter', e => { if(!acceptsDrag(e))return; e.preventDefault(); if (!dragging++) { beforeDrag=phase; $('full-drop').hidden=false; setState('dragover'); } });
document.addEventListener('dragover', e => { if(acceptsDrag(e)) { e.preventDefault(); e.dataTransfer.dropEffect='copy'; } });
document.addEventListener('dragleave', e => { if (dragging && (--dragging === 0 || !e.relatedTarget)) { dragging=0; $('full-drop').hidden=true; setState(beforeDrag); } });
document.addEventListener('drop', e => {
  if(!acceptsDrag(e))return; e.preventDefault(); dragging=0; $('full-drop').hidden=true;
  handleInput('drop', e.dataTransfer.files.length ? e.dataTransfer.files : e.dataTransfer.getData('text/uri-list').split('\n').find(s=>!s.startsWith('#')) || e.dataTransfer.getData('text/plain'));
});
function layout() {
  const box = surface.getBoundingClientRect(); if (!box.width) return;
  width=box.width; height=box.height;
  const mobile=width<600, active=!!current || ['result','error','unsupported'].includes(phase);
  radius=mobile ? Math.min(width*.405,(height-90)/2) : Math.min(width*.145, height*.36);
  if (active && !mobile) radius=Math.min(radius,height*.245);
  if (phase==='dragover') radius=Math.min(radius*1.04,width*.45,height*.46);
  art.setAttribute('viewBox',`0 0 ${width} ${height}`);
  const word=$('word-shape'); word.setAttribute('x',width/2); word.setAttribute('y',active ? (mobile ? 70 : height*.46) : height*.63);
  const logoSize=width*(active ? .30 : .36);
  word.removeAttribute('textLength');word.removeAttribute('lengthAdjust');
  word.setAttribute('font-size',logoSize);word.setAttribute('letter-spacing',-logoSize*.045);
  if (active || phase==='dragover' || reduced.matches) { tx=width/2; ty=active ? height-radius-12 : height/2; }
  else { tx=width/2; ty=height/2; }
  x=Math.max(radius,Math.min(width-radius,x)); y=Math.max(radius,Math.min(height-radius,y));
  const wordGroup=document.querySelector('.word-outside');
  if(textMode) {
    // One uniform transform keeps the original letterforms intact as the logo docks.
    const logoScale=mobile?.52:.34;
    const originalY=Number(word.getAttribute('y'));
    const targetY=mobile?57:91;
    wordGroup.style.transform=`translate(${width/2*(1-logoScale)}px,${targetY-originalY*logoScale}px) scale(${logoScale})`;
  } else wordGroup.style.transform='none';
  lens.style.width=lens.style.height=`${radius*2}px`;
  $('photo-deck').style.setProperty('--deck-step',`${radius*1.8/Math.max(1,Math.min(5,attachments.length)-1)}px`);
  const textFrame=$('text-frame');
  textFrame.style.setProperty('--morph-x',String(radius*2/Math.max(1,textFrame.offsetWidth)));
  textFrame.style.setProperty('--morph-y',String(radius*2/Math.max(1,textFrame.offsetHeight)));
  textFrame.style.setProperty('--morph-offset',`${ty-(textFrame.offsetTop+textFrame.offsetHeight/2)}px`);
  $('clear-word').setAttribute('visibility', active ? 'hidden':'visible');
  $('core-layer').setAttribute('visibility', active ? 'hidden':'visible');
  surface.style.setProperty('--outside-blur',`${Math.max(2.6,Math.min(8,width*.008))}px`);
}
function draw(px,py,r) {
  $('clip-circle').setAttribute('cx',px); $('clip-circle').setAttribute('cy',py); $('clip-circle').setAttribute('r',r);
  lens.style.transform=`translate3d(${px-r}px,${py-r}px,0) scale(${r/radius})`;
  lens.style.transformOrigin='top left';
  $('core-circle').setAttribute('cx',px); $('core-circle').setAttribute('cy',py); $('core-circle').setAttribute('r',r*.9);
  const zoom = reduced.matches ? 1.07 : 1.10;
  $('clear-word').setAttribute('transform',`translate(${px} ${py}) scale(1.17) translate(${-px} ${-py})`);
  $('core-word').setAttribute('transform',`translate(${px} ${py}) scale(${zoom}) translate(${-px} ${-py})`);
  lens.style.setProperty('--tilt-x',`${tiltX}deg`); lens.style.setProperty('--tilt-y',`${tiltY}deg`);
  lens.style.setProperty('--light-x',`${35+tiltY*3}%`); lens.style.setProperty('--light-y',`${23-tiltX*3}%`);
  lens.style.setProperty('--shadow-x',`${-tiltY*1.1}px`);
  lens.style.setProperty('--reflection-angle',`${135+tiltY*5}deg`);
  const image=$('lens-image'); image.setAttribute('x',px-r); image.setAttribute('y',py-r); image.setAttribute('width',2*r); image.setAttribute('height',2*r);
  $('lens-excerpt').setAttribute('x',px); $('lens-excerpt').setAttribute('y',py);
}
function tick(now) {
  frame=0;
  const dt=Math.min(.04,Math.max(.001,(now-(lastTime||now-16))/1000)); lastTime=now;
  if(document.hidden || document.body.dataset.route!=='home')return;
  if(!introDone && !current) {
    const t=Math.min((now-introStart)/1800,1);
    const a=t<.35 ? t/.35 : (t-.35)/.65;
    x=t<.35 ? width*.91+(width*.23-width*.91)*a : width*.23+(width*.5-width*.23)*a;
    y=height*.5; draw(x,y,radius*Math.min(1,.09+t*3));
    if(t===1) {introDone=true;stage.dataset.intro='false';openInput();}
  } else {
    if(touch && phase==='idle' && !current && !reduced.matches) {tx=width/2+Math.sin(now/1800)*9;ty=height/2+Math.cos(now/2100)*7;}
    if(reduced.matches) {x=tx;y=ty;vx=vy=tiltX=tiltY=0;}
    else {
      // Critically damped spring, integrated in seconds: identical feel at 60/120 Hz.
      const omega=12, decay=Math.exp(-omega*dt);
      const dx=x-tx,dy=y-ty, ax=vx+omega*dx,ay=vy+omega*dy;
      x=tx+(dx+ax*dt)*decay; y=ty+(dy+ay*dt)*decay;
      vx=(vx-omega*ax*dt)*decay;vy=(vy-omega*ay*dt)*decay;
      const ease=1-Math.exp(-9*dt);
      tiltX+=(Math.max(-7,Math.min(7,-vy*.018))-tiltX)*ease;
      tiltY+=(Math.max(-7,Math.min(7,vx*.018))-tiltY)*ease;
    }
    draw(x,y,radius);
  }
  if(!reduced.matches && (!introDone || touch || Math.abs(tx-x)+Math.abs(ty-y)+Math.abs(tiltX)+Math.abs(tiltY)>.03)) frame=requestAnimationFrame(tick);
}
function startMotion(){if(!frame && !document.hidden)frame=requestAnimationFrame(tick);}
surface.addEventListener('pointermove',e=>{if(touch||current||phase!=='idle'||reduced.matches||!introDone)return;const b=surface.getBoundingClientRect();tx=Math.max(radius,Math.min(width-radius,e.clientX-b.left));ty=Math.max(radius,Math.min(height-radius,e.clientY-b.top));startMotion();});
surface.addEventListener('pointerleave',()=>{if(!current){tx=width/2;ty=height/2;startMotion();}});
new ResizeObserver(()=>{layout();startMotion();}).observe(surface);
document.addEventListener('visibilitychange',()=>{document.documentElement.dataset.paused=String(document.hidden);if(document.hidden){cancelAnimationFrame(frame);frame=0;lastTime=0;}else startMotion();});
document.addEventListener('fakit:route',()=>{layout();startMotion();});
reduced.addEventListener('change',()=>{introDone=true;stage.dataset.intro='false';openInput();layout();startMotion();});
$('lens-ready').hidden=!$('model-status').classList.contains('ready');
new MutationObserver(()=>{ $('lens-ready').hidden=!$('model-status').classList.contains('ready'); }).observe($('model-status'),{attributes:true,attributeFilter:['class']});
addEventListener('pagehide',()=>{abortRun();releasePreview();cancelAnimationFrame(frame);frame=0;});
const fragment=new URLSearchParams(location.hash.slice(1));
if(fragment.has('image')||fragment.has('import_error')) {
  history.replaceState(null,'',location.pathname+location.search+'#/');
  if(fragment.has('image')) handleInput('url',fragment.get('image'));
  else setState('error','이 주소는 가져올 수 없어요. 이미지를 저장한 뒤 직접 넣어 주세요.');
}
layout();startMotion();

if(reduced.matches && !current)openInput();
