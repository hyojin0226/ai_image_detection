"use strict";
const $=id=>document.getElementById(id);
let mode="file",ready=false,busy=false,connecting=false,selectedFile=null,fileReady=false,previewUrl="",selection=0,result=null,detailsRendered=false,abort=null;
let enginePromise=null;
let notebookBase="",notebookOnline=false,probing=false;
const engine=()=>enginePromise||(enginePromise=import('/browser-engine.js'));
function error(message){$("error").textContent=message;$("error").hidden=!message;}
function refresh(){ $("analyze").disabled=busy||(mode==="file"?(!ready||!fileReady):(!notebookOnline||!$("video-url").value.trim())); }
function status(message){$("progress-message").textContent=message;const m=message.match(/Grad-CAM\s+(\d+)\/(\d+)/);if(m){$("frame-progress").hidden=false;$("frame-progress").max=Number(m[2]);$("frame-progress").value=Number(m[1]);}}
async function connect(){
  if(connecting)return;connecting=true;$("reconnect").hidden=true;
  try{const module=await engine();await module.prepare(message=>{$("connection").textContent=message;});ready=true;$("connection").textContent="● 이 브라우저에서 분석 준비 완료";$("connection").className="status online";}
  catch(cause){ready=false;$("connection").textContent="● 모델 준비 실패";$("connection").className="status offline";$("reconnect").hidden=false;error(cause.message);}
  connecting=false;refresh();
}
$("reconnect").addEventListener("click",connect);
function tab(value){mode=value;for(const name of ["file","url"]){$(name+"-tab").setAttribute("aria-selected",String(name===mode));$(name+"-panel").hidden=name!==mode;}error("");refresh();}
for(const name of ["file","url"]){
 $(name+"-tab").addEventListener("click",()=>tab(name));
 $(name+"-tab").addEventListener("keydown",event=>{if(["ArrowLeft","ArrowRight","Home","End"].includes(event.key)){event.preventDefault();const next=event.key==="Home"?"file":event.key==="End"?"url":mode==="file"?"url":"file";tab(next);$(next+"-tab").focus();}});
}
function chooseFile(file){
 if(busy)return;const current=++selection;error("");selectedFile=file||null;fileReady=false;
 if(previewUrl)URL.revokeObjectURL(previewUrl);
 const video=$("video-preview");video.onloadedmetadata=null;video.onerror=null;video.removeAttribute("src");video.load();
 $("preview").hidden=!file;$("filename").textContent=file?`${file.name} · ${(file.size/1024**2).toFixed(1)} MB`:"선택한 파일이 없습니다.";refresh();
 if(!file)return;
 if(!file.size||file.size>90*1024**2){error("90 MB 이하의 영상 파일을 선택하세요.");$("preview").hidden=true;return;}
 $("preview-name").textContent=file.name;$("preview-meta").textContent="영상 길이를 확인하고 있습니다…";
 video.onloadedmetadata=()=>{if(current!==selection)return;const d=video.duration;
  if(!Number.isFinite(d)||d<3||d>40){error(`3~40초 영상을 선택하세요. 선택한 영상: ${Number.isFinite(d)?d.toFixed(1)+"초":"길이 확인 불가"}`);fileReady=false;$("preview-meta").textContent="지원하는 길이의 영상을 선택해 주세요.";}
  else{fileReady=true;const planned=d<=10?16:d<=20?32:d<=30?48:64;$("preview-meta").textContent=`${d.toFixed(1)}초 · ${video.videoWidth} × ${video.videoHeight} · ${planned}프레임 예정 · 이 브라우저에서 분석`;}
  refresh();
 };
 video.onerror=()=>{if(current!==selection)return;fileReady=false;$("preview-meta").textContent="이 브라우저에서 재생할 수 없는 형식입니다. MP4(H.264) 또는 WebM 파일을 선택하세요.";error("브라우저가 재생할 수 있는 영상 파일이 필요합니다.");refresh();};
 previewUrl=URL.createObjectURL(file);video.src=previewUrl;
}
$("video-file").addEventListener("change",()=>chooseFile($("video-file").files[0]));
$("remove-file").addEventListener("click",()=>{$("video-file").value="";chooseFile(null);});
$("video-url").addEventListener("input",refresh);
for(const name of ["dragenter","dragover"])$("drop-zone").addEventListener(name,event=>{event.preventDefault();if(!busy)$("drop-zone").classList.add("dragging");});
for(const name of ["dragleave","drop"])$("drop-zone").addEventListener(name,event=>{event.preventDefault();$("drop-zone").classList.remove("dragging");});
$("drop-zone").addEventListener("drop",event=>{if(busy)return;if(event.dataTransfer.files.length!==1){error("영상을 한 개씩 선택하세요.");return;}chooseFile(event.dataTransfer.files[0]);});
function frameCard(frame){
 const card=document.createElement("article");card.className="frame";const button=document.createElement("button");button.type="button";
 button.setAttribute("aria-label",`${frame.index}번 프레임 ${frame.timestamp_sec.toFixed(3)}초 확대`);
 const image=document.createElement("img");image.loading="lazy";image.src=frame.comparison_url;image.alt=`${frame.index}번 프레임 원본과 Grad-CAM`;button.append(image);card.append(button);
 const caption=document.createElement("div");caption.className="caption";const title=document.createElement("strong");title.textContent=`${frame.index}번 · ${frame.timestamp_sec.toFixed(3)}초`;
 const text=document.createElement("p");text.textContent=`구간 ${frame.segment} · AI 점수 영향 ${frame.video_score_delta>=0?"+":""}${frame.video_score_delta.toFixed(5)}\n강조 영역: ${frame.strongest_zone||"양의 신호 없음"}`;caption.append(title,text);
 if(frame.note){const note=document.createElement("p");note.className="frame-note";note.textContent=frame.note;caption.append(note);}card.append(caption);
 button.addEventListener("click",()=>{$("zoom-title").textContent=`${frame.index}번 · ${frame.timestamp_sec.toFixed(3)}초 · 원본 / Grad-CAM`;$("zoom-image").src=frame.comparison_url;$("zoom-note").textContent=`${frame.target_class==="fake"?"AI 생성":"실제 촬영"} 판단을 지지하는 양의 신호입니다. 합성 위치의 확정 표시가 아닙니다.`;$("zoom").showModal();});return card;
}
$("close-zoom").addEventListener("click",()=>$("zoom").close());
function display(value){
 result=value;detailsRendered=false;$("result-title").textContent=result.prediction_ko;$("score").textContent=result.fake_score.toFixed(4);
 $("result-meta").textContent=`${result.duration_sec.toFixed(1)}초 영상 · 분석 프레임 ${result.analyzed_frames}/${result.planned_frames} · ${result.inference_location==="notebook"?"노트북 서버":"브라우저"} 분석 ${result.elapsed_sec}초`+(result.skipped_segments.length?`\n미분석 구간 ${result.skipped_segments.length}개`:"");
 $("target-label").textContent=`강조 대상: ${result.prediction==="fake"?"AI 생성":"실제 촬영"}`;
 const map=new Map(result.frames.map(f=>[f.index,f]));$("main-frames").replaceChildren(...result.main_frame_indices.map(i=>frameCard(map.get(i))));
 $("detail-frames").replaceChildren();$("details").open=false;$("details-title").textContent=`세부사항 · 나머지 ${result.detail_frame_indices.length}개`;
 $("segments").replaceChildren(...result.segments.map(s=>{const span=document.createElement("span");span.className="segment";span.textContent=`${s.start_sec.toFixed(1)}~${s.end_sec.toFixed(1)}초 · ${s.fake_score.toFixed(3)}`;return span;}));
 $("result").hidden=false;$("result").scrollIntoView({behavior:matchMedia("(prefers-reduced-motion: reduce)").matches?"instant":"smooth",block:"start"});
}
$("details").addEventListener("toggle",()=>{if(!result||!$("details").open||detailsRendered)return;detailsRendered=true;const map=new Map(result.frames.map(f=>[f.index,f]));$("detail-frames").replaceChildren(...result.detail_frame_indices.map(i=>frameCard(map.get(i))));});
function youtubeURL(value){
 const url=new URL(value);if(url.protocol!=="https:"||url.username||url.password||!["youtube.com","www.youtube.com","m.youtube.com","youtu.be"].includes(url.hostname))throw Error("공개 YouTube Shorts 주소를 입력하세요.");
 const id=url.hostname==="youtu.be"?url.pathname.slice(1):url.pathname.startsWith("/shorts/")?url.pathname.slice(8).replace(/\/$/,""):url.pathname==="/watch"?url.searchParams.get("v"):"";
 if(!/^[\w-]{11}$/.test(id||""))throw Error("개별 YouTube 영상 주소를 입력하세요.");return `https://www.youtube.com/watch?v=${id}`;
}
function validNotebook(value){
 try{const u=new URL(value);return u.protocol==="https:"&&/^[a-z0-9-]+\.trycloudflare\.com$/.test(u.hostname)&&!u.port&&!u.username&&!u.password&&u.pathname==="/"&&!u.search&&!u.hash?u.origin:"";}catch{return "";}
}
function notebookStatus(online,base=""){
 notebookOnline=online;notebookBase=online?base:"";
 $("notebook-status").textContent=online?"● 노트북 서버 연결됨 · 주소 분석 가능":"● 노트북 서버 꺼짐 또는 연결 안 됨 · 영상 파일 분석을 이용하세요";
 $("notebook-status").className=online?"status online":"status offline";refresh();
}
async function checkNotebook(){
 if(probing)return notebookOnline;probing=true;
 try{
  const config=await fetch('/api/config',{cache:'no-store',signal:AbortSignal.timeout(5000)});
  if(!config.ok)throw Error();const base=validNotebook((await config.json()).notebookApiBase);
  if(!base)throw Error();
  const response=await fetch(base+'/health',{cache:'no-store',signal:AbortSignal.timeout(5000)});
  if(!response.ok)throw Error();const health=await response.json();
  if(health.status!=="online"||health.mode!=="segmented64_v1"||health.frame_policy!=="duration_v1")throw Error();
  notebookStatus(true,base);
 }catch{notebookStatus(false);}finally{probing=false;}
 return notebookOnline;
}
$("check-notebook").addEventListener("click",checkNotebook);
window.addEventListener("focus",()=>{if(!busy)checkNotebook();});
setInterval(()=>{if(!busy&&!document.hidden)checkNotebook();},15000);
function pause(milliseconds,signal){return new Promise((resolve,reject)=>{
 if(signal.aborted)return reject(new DOMException("Cancelled","AbortError"));
 const cancel=()=>{clearTimeout(timer);reject(new DOMException("Cancelled","AbortError"));};
 const timer=setTimeout(()=>{signal.removeEventListener("abort",cancel);resolve();},milliseconds);
 signal.addEventListener("abort",cancel,{once:true});
});}
async function notebookAnalysis(url,signal){
 status("노트북 서버 연결을 확인하고 있습니다…");
 if(!await checkNotebook())throw Error("노트북 서버가 꺼져 있거나 연결되지 않았습니다. 서버를 켜거나 영상 파일 분석을 이용하세요.");
 const base=notebookBase;
 status("노트북에서 공개 YouTube 영상을 가져와 길이에 맞는 프레임을 분석합니다.");
 const response=await fetch(base+'/jobs/youtube',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({url}),signal});
 const job=await response.json();if(!response.ok)throw Error(job.error||"주소 분석을 시작하지 못했습니다.");
 if(!/^[A-Za-z0-9_-]{20,80}$/.test(job.id||""))throw Error("올바르지 않은 작업 응답입니다.");
 const path='/jobs/'+job.id;const started=performance.now();
 while(performance.now()-started<15*60*1000){
  await pause(1500,signal);
  const reply=await fetch(base+path,{cache:'no-store',signal});
  const state=await reply.json();if(!reply.ok)throw Error(state.error||"분석 결과를 가져오지 못했습니다.");
  status(state.message||"노트북에서 분석 중입니다.");
  if(state.state==="failed")throw Error(state.message||"노트북에서 분석하지 못했습니다.");
  if(state.state==="completed"){
   const value=state.result;
   if(value.frame_policy!=="duration_v1"||![16,32,48,64].includes(value.planned_frames))throw Error("노트북 서버를 새 버전으로 다시 켜 주세요.");
   for(const frame of value.frames)for(const field of ['original_url','comparison_url','gradcam_url']){
    if(!frame[field])continue;
    if(!frame[field].startsWith(path+'/frames/')||!/^[A-Za-z0-9_.-]+\.png$/.test(frame[field].slice((path+'/frames/').length)))throw Error("잘못된 프레임 주소입니다.");
    frame[field]=base+frame[field];
   }
   value.inference_location="notebook";value.source={kind:'youtube',url};return value;
  }
 }
 throw Error("분석 대기 시간이 지났습니다. 노트북 서버 상태를 확인하세요.");
}
$("cancel-analysis").addEventListener("click",()=>abort?.abort());
$("analysis-form").addEventListener("submit",async event=>{
 event.preventDefault();if(busy||(mode==="file"&&!ready))return;error("");let timer;const inputMode=mode;
 try{
  if(mode==="file"&&(!selectedFile||!fileReady))throw Error("지원하는 영상 파일을 선택하세요.");
  const url=mode==="url"?youtubeURL($("video-url").value.trim()):null;
  busy=true;abort=new AbortController();refresh();$("result").hidden=true;$("progress").hidden=false;$("cancel-analysis").hidden=false;$("frame-progress").hidden=true;
  for(const id of ["video-file","video-url","remove-file","file-tab","url-tab"])$(id).disabled=true;
  $("progress-title").textContent=url?"노트북에서 길이별 프레임 분석":"이 브라우저에서 길이별 프레임 분석";$("elapsed").textContent="분석을 준비하고 있습니다.";const start=performance.now();
  status("영상 길이와 분석할 프레임을 준비하고 있습니다.");
  timer=setInterval(()=>{$("elapsed").textContent=`경과 ${Math.floor((performance.now()-start)/1000)}초 · 분석 중에는 이 탭을 열어 두세요.`;},1000);
  const value=url?await notebookAnalysis(url,abort.signal):await(await engine()).analyzeFile(selectedFile,{progress:status,signal:abort.signal});
  display(value);
 }catch(cause){
  error(cause.name==="AbortError"?(inputMode==="url"?"결과 대기를 취소했습니다. 노트북에서 시작된 작업은 완료될 수 있습니다.":"분석을 취소했습니다."):inputMode==="url"&&cause instanceof TypeError?"노트북 서버 연결이 끊겼습니다. 서버와 인터넷 연결을 확인하거나 영상 파일 분석을 이용하세요.":cause.message||"분석하지 못했습니다.");
  if(inputMode==="url")checkNotebook();
 }
 finally{clearInterval(timer);busy=false;abort=null;$("progress").hidden=true;$("cancel-analysis").hidden=true;for(const id of ["video-file","video-url","remove-file","file-tab","url-tab"])$(id).disabled=false;refresh();}
});
connect();
checkNotebook();
