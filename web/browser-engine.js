import {pooled,head,featureGradient,loadHeads} from './video-math.js';
import {inference} from './inference-client.js?v=classifiers';
import {MODEL_BASE} from './model-release.js';
let loading=null;
document.addEventListener('fakit:inference-released',()=>{loading=null;});
const next=()=>new Promise(resolve=>{const timer=setTimeout(resolve,100);requestAnimationFrame(()=>{clearTimeout(timer);resolve();});});
const check=signal=>{if(signal?.aborted)throw new DOMException('분석을 취소했습니다.','AbortError');};
async function json(url){const r=await fetch(url);if(!r.ok)throw Error('모델 파일을 불러오지 못했습니다.');return r.json();}
export function prepare(progress=()=>{}){
  if(loading)return loading;
  loading=(async()=>{
    progress('이 브라우저에서 분석할 모델을 준비하고 있습니다.');
    const {config,backend}=await inference('video-init');
    const members=await loadHeads(config,MODEL_BASE);
    return {members,config,backend};
  })().catch(error=>{loading=null;throw error;});
  return loading;
}
function waitEvent(target,name,signal,timeout=15000){
  return new Promise((resolve,reject)=>{
    let timer;
    const clean=()=>{clearTimeout(timer);target.removeEventListener(name,ok);target.removeEventListener('error',bad);signal?.removeEventListener('abort',abort);};
    const ok=()=>{clean();resolve();};
    const bad=()=>{clean();reject(Error('영상을 읽지 못했습니다. 브라우저에서 재생 가능한 MP4 또는 WebM 파일을 선택하세요.'));};
    const abort=()=>{clean();reject(new DOMException('분석을 취소했습니다.','AbortError'));};
    target.addEventListener(name,ok,{once:true});target.addEventListener('error',bad,{once:true});signal?.addEventListener('abort',abort,{once:true});
    timer=setTimeout(bad,timeout);if(signal?.aborted)abort();
  });
}
function canvas(width,height){const c=document.createElement('canvas');c.width=width;c.height=height;return c;}
function rawCanvas(rgb){const c=canvas(256,256),ctx=c.getContext('2d');const data=ctx.createImageData(256,256);
  for(let p=0;p<256*256;p++){data.data[p*4]=rgb[p*3];data.data[p*4+1]=rgb[p*3+1];data.data[p*4+2]=rgb[p*3+2];data.data[p*4+3]=255;}
  ctx.putImageData(data,0,0);return c;}
async function openVideo(file,signal){
  const video=document.createElement('video');video.preload='auto';video.muted=true;video.playsInline=true;
  const url=URL.createObjectURL(file);const loaded=waitEvent(video,'loadedmetadata',signal);video.src=url;
  try{await loaded;}catch(error){URL.revokeObjectURL(url);throw error;}
  if(!Number.isFinite(video.duration)||video.duration<3||video.duration>40){video.removeAttribute('src');video.load();URL.revokeObjectURL(url);throw Error('3~40초 영상을 선택하세요.');}
  const c=canvas(256,256),ctx=c.getContext('2d',{willReadFrequently:true});
  const ratio=Math.min(256/video.videoWidth,256/video.videoHeight),w=Math.round(video.videoWidth*ratio),h=Math.round(video.videoHeight*ratio);
  async function capture(timestamp){
    check(signal);const seek=waitEvent(video,'seeked',signal);video.currentTime=timestamp;await seek;
    if(video.readyState<2)await waitEvent(video,'loadeddata',signal);
    ctx.fillStyle='black';ctx.fillRect(0,0,256,256);ctx.imageSmoothingEnabled=true;ctx.imageSmoothingQuality='high';
    ctx.drawImage(video,(256-w)>>1,(256-h)>>1,w,h);
    const rgba=ctx.getImageData(0,0,256,256).data,rgb=new Uint8Array(256*256*3);let black=true;
    for(let p=0;p<256*256;p++)for(let k=0;k<3;k++){const value=rgba[p*4+k];rgb[p*3+k]=value;if(value>3)black=false;}
    return {rgb,timestamp_sec:timestamp,black};
  }
  return {video,capture,close(){video.pause();video.removeAttribute('src');video.load();URL.revokeObjectURL(url);}};
}
async function segments(reader,progress,signal){
  const duration=reader.video.duration,groups=[],black=[],skipped=[];
  for(let s=0;s<8;s++){
    const start=s*duration/8,end=(s+1)*duration/8,candidates=[];
    for(let j=0;j<8;j++){
      check(signal);const f=await reader.capture((s*8+j+.5)*duration/64);
      if(f.black)black.push(f.timestamp_sec);else candidates.push(f);
      progress(`프레임 추출 ${s*8+j+1}/64`);
    }
    if(candidates.length<8)for(let j=0;j<16;j++){
      const f=await reader.capture(start+(j+.5)*(end-start)/16);
      if(f.black)black.push(f.timestamp_sec);else candidates.push(f);
    }
    if(candidates.length<8){skipped.push({segment:s+1,start_sec:start,end_sec:end,reason:'검은 화면 제외 후 유효 시점 8개를 확보하지 못해 미분석'});continue;}
    candidates.sort((a,b)=>a.timestamp_sec-b.timestamp_sec);
    const chosen=Array.from({length:8},(_,i)=>candidates[Math.floor((i+.5)*candidates.length/8)]);
    groups.push({segment:s+1,start_sec:start,end_sec:end,frames:chosen});
  }
  if(!groups.length)throw Error('검은 화면 제외 후 분석할 구간이 없습니다.');
  return {groups,black,skipped};
}
function heatmap(activation,gradient){
  const mean=new Float64Array(1280);
  for(let p=0;p<64;p++)for(let c=0;c<1280;c++)mean[c]+=gradient[p*1280+c]/64;
  const heat=new Float64Array(64);let max=0;
  for(let p=0;p<64;p++){let sum=0;for(let c=0;c<1280;c++)sum+=activation[p*1280+c]*mean[c];heat[p]=Math.max(0,sum);max=Math.max(max,heat[p]);}
  if(max<=1e-20)return null;
  return heat.map(v=>v/max);
}
function render(rgb,heat,target){
  const original=rawCanvas(rgb),overlay=canvas(256,256),ctx=overlay.getContext('2d'),image=ctx.createImageData(256,256);
  const full=new Float64Array(256*256);let x0=256,y0=256,x1=0,y1=0;
  for(let y=0;y<256;y++)for(let x=0;x<256;x++){
    const p=y*256+x,at=p*3,mask=rgb[at]||rgb[at+1]||rgb[at+2];let h=0;
    if(mask){x0=Math.min(x0,x);x1=Math.max(x1,x+1);y0=Math.min(y0,y);y1=Math.max(y1,y+1);}
    if(heat&&mask){const fy=Math.max(0,Math.min(7,(y+.5)/32-.5)),fx=Math.max(0,Math.min(7,(x+.5)/32-.5));
      const a=Math.floor(fy),b=Math.min(7,a+1),c=Math.floor(fx),d=Math.min(7,c+1),dy=fy-a,dx=fx-c;
      h=(heat[a*8+c]*(1-dx)+heat[a*8+d]*dx)*(1-dy)+(heat[b*8+c]*(1-dx)+heat[b*8+d]*dx)*dy;
    }
    full[p]=h;const alpha=.6*h,color=[255,255*(1-h),0];
    for(let k=0;k<3;k++)image.data[p*4+k]=Math.round(rgb[at+k]*(1-alpha)+color[k]*alpha);
    image.data[p*4+3]=255;
  }
  ctx.putImageData(image,0,0);
  let zone=null,max=0;
  for(let row=0;row<3;row++)for(let col=0;col<3;col++){
    let mass=0;for(let y=y0+Math.floor((y1-y0)*row/3);y<y0+Math.floor((y1-y0)*(row+1)/3);y++)
      for(let x=x0+Math.floor((x1-x0)*col/3);x<x0+Math.floor((x1-x0)*(col+1)/3);x++)mass+=full[y*256+x];
    if(mass>0&&mass>=max){max=mass;zone=[['왼쪽 위','위쪽 중앙','오른쪽 위'],['왼쪽 중앙','중앙','오른쪽 중앙'],['왼쪽 아래','아래쪽 중앙','오른쪽 아래']][row][col];}
  }
  const comparison=canvas(512,280),draw=comparison.getContext('2d');draw.fillStyle='white';draw.fillRect(0,0,512,280);
  draw.fillStyle='black';draw.font='12px sans-serif';draw.fillText('Original',6,16);draw.fillText(heat?`Grad-CAM: ${target==='fake'?'AI':'REAL'}`:'No positive signal',262,16);
  draw.drawImage(original,0,24);draw.drawImage(heat?overlay:original,256,24);
  return {comparison_url:comparison.toDataURL('image/png'),strongest_zone:zone, heatmap:heat ? Array.from(heat) : null};
}
export async function analyzeFile(file,{progress=()=>{},signal}={}){
  if(!file?.size||file.size>90*1024**2)throw Error('90 MB 이하의 영상을 선택하세요.');
  const started=performance.now(),engine=await prepare(progress);check(signal);
  const reader=await openVideo(file,signal),duration=reader.video.duration;let selected;
  try{selected=await segments(reader,progress,signal);}finally{reader.close();}
  const {groups,black,skipped}=selected,records=[],statistics=[],scores=[];
  for(const group of groups){
    group.sequence=new Float64Array(8*257);
    for(let i=0;i<8;i++){
      check(signal);const f=group.frames[i];
      const [activation,embedding,real]=await inference('video-features',{rgb:f.rgb});
      f.activation=activation;group.sequence.set(embedding,i*257);group.sequence[i*257+256]=1-real[0];
      progress(`구간 ${group.segment}/8 · 프레임 ${i+1}/8 특징 분석`);await next();
    }
    const x=pooled(group.sequence),prediction=head(x,engine.members,true);
    statistics.push(x);scores.push(prediction.score);group.featureGradient=featureGradient(group.sequence,x,prediction.gradient);
  }
  const score=scores.reduce((a,b)=>a+b,0)/groups.length,target=score>=.5?'fake':'real';
  for(let g=0;g<groups.length;g++){
    const group=groups[g],mean=new Float64Array(257);
    for(let i=0;i<8;i++)for(let j=0;j<257;j++)mean[j]+=group.sequence[i*257+j]/8;
    for(let i=0;i<8;i++){
      check(signal);const f=group.frames[i],changed=group.sequence.slice();changed.set(mean,i*257);
      const alternative=score+(head(pooled(changed),engine.members).score-scores[g])/groups.length;
      const weights=new Float32Array(257),direction=(target==='fake'?1:-1)/groups.length;
      for(let j=0;j<257;j++)weights[j]=group.featureGradient[i*257+j]*direction;
      try{
        const gradient=await inference('video-gradient',{activation:f.activation,weights});
        const heat=heatmap(f.activation,gradient);
        const rendered=render(f.rgb,heat,target);
        records.push({index:records.length+1,timestamp_sec:f.timestamp_sec,segment:group.segment,
          video_score_delta:score-alternative,replacement_video_score:alternative,
          image_fake_score:group.sequence[i*257+256],target_class:target,gradcam_available:!!heat,
          note:heat?null:'양의 Grad-CAM 신호가 없어 색을 표시하지 않습니다.',...rendered});
      }finally{f.activation=null;}
      progress(`Grad-CAM ${records.length}/${groups.length*8}`);await next();
    }
  }
  const main=[...records].sort((a,b)=>Math.abs(b.video_score_delta)-Math.abs(a.video_score_delta)||a.timestamp_sec-b.timestamp_sec||a.index-b.index).slice(0,8).map(f=>f.index);
  return {status:'completed',model:'v1.0.0-browser',analysis_mode:'browser_segmented64_v1',threshold:.5,
    inference_location:'browser',backend:engine.backend,source:{kind:'file',title:file.name},duration_sec:duration,
    fake_score:score,prediction:target,prediction_ko:target==='fake'?'AI 생성 영상 추정':'실제 촬영 영상 추정',
    thumbnail:rawCanvas(groups[0].frames[0].rgb).toDataURL('image/jpeg',.72),
    planned_frames:64,analyzed_frames:records.length,segments_used:groups.length,black_seek_positions_sec:black,skipped_segments:skipped,
    frames:records,main_frame_indices:main,detail_frame_indices:records.filter(f=>!main.includes(f.index)).map(f=>f.index),
    segments:groups.map((group,i)=>({segment:group.segment,start_sec:group.start_sec,end_sec:group.end_sec,fake_score:scores[i]})),
    elapsed_sec:Math.round((performance.now()-started)/100)/10,accuracy_validated_for_this_mode:false,score_is_calibrated_probability:false,
    limitations_ko:'브라우저 디코딩·크기 조정은 FFmpeg와 차이가 있을 수 있으며 브라우저 방식의 정확도는 별도 검증 전입니다. Grad-CAM은 선택한 분류의 신호이며 합성 사실의 증명이 아닙니다.'};
}
