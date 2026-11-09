// Same frozen float64 temporal statistics and two RBF heads. No model fitting.
const WIDTH=257,DIM=256,SIZE=772;
export function pooled(sequence){
  if(sequence.length!==8*WIDTH)throw Error('구간 특징 크기가 올바르지 않습니다.');
  const x=new Float64Array(SIZE);
  for(let j=0;j<DIM;j++){
    let mean=0;for(let t=0;t<8;t++)mean+=sequence[t*WIDTH+j]/8;
    let variance=0,difference=0;
    for(let t=0;t<8;t++)variance+=(sequence[t*WIDTH+j]-mean)**2/8;
    for(let t=1;t<8;t++)difference+=Math.abs(sequence[t*WIDTH+j]-sequence[(t-1)*WIDTH+j])/7;
    x[j]=mean;x[DIM+j]=Math.sqrt(variance);x[DIM*2+j]=difference;
  }
  let mean=0,min=Infinity,max=-Infinity;
  for(let t=0;t<8;t++){const s=sequence[t*WIDTH+DIM];mean+=s/8;min=Math.min(min,s);max=Math.max(max,s);}
  let variance=0;for(let t=0;t<8;t++)variance+=(sequence[t*WIDTH+DIM]-mean)**2/8;
  x[768]=mean;x[769]=Math.sqrt(variance);x[770]=min;x[771]=max;
  return x;
}
export function head(x,members,withGradient=false){
  let score=0;const gradient=withGradient?new Float64Array(SIZE):null;
  for(const m of members){
    const z=new Float64Array(SIZE),norm=new Float64Array(SIZE),dz=withGradient?new Float64Array(SIZE):null;
    for(let k=0;k<SIZE;k++){norm[k]=(x[k]-m.mean[k])/m.scale[k];z[k]=Math.max(-10,Math.min(10,norm[k]))*m.block[k];}
    let residual=0;
    for(let r=0;r<m.coefficient.length;r++){
      let distance=0;const offset=r*SIZE;
      for(let k=0;k<SIZE;k++)distance+=(z[k]-m.support[offset+k])**2;
      const weighted=Math.exp(-m.gamma*distance)*m.coefficient[r];residual+=weighted;
      if(withGradient){const c=-2*m.gamma*weighted;for(let k=0;k<SIZE;k++)dz[k]+=c*(z[k]-m.support[offset+k]);}
    }
    const raw=4*residual,p=1/(1+Math.exp(-Math.max(-60,Math.min(60,raw))));score+=p/members.length;
    if(withGradient&&raw>=-60&&raw<=60){const factor=4*p*(1-p)/members.length;
      for(let k=0;k<SIZE;k++)if(norm[k]>=-10&&norm[k]<=10)gradient[k]+=factor*dz[k]*m.block[k]/m.scale[k];}
  }
  return {score,gradient};
}
export function featureGradient(sequence,statistics,g){
  const out=new Float64Array(8*WIDTH);
  for(let j=0;j<DIM;j++)for(let t=0;t<8;t++){
    const at=t*WIDTH+j,std=statistics[DIM+j];
    let value=g[j]/8+(std>0?g[DIM+j]*(sequence[at]-statistics[j])/(8*std):0);
    if(t>0)value+=g[DIM*2+j]*Math.sign(sequence[at]-sequence[at-WIDTH])/7;
    if(t<7)value-=g[DIM*2+j]*Math.sign(sequence[at+WIDTH]-sequence[at])/7;
    out[at]=value;
  }
  let minCount=0,maxCount=0;
  for(let t=0;t<8;t++){const s=sequence[t*WIDTH+DIM];if(s===statistics[770])minCount++;if(s===statistics[771])maxCount++;}
  for(let t=0;t<8;t++){
    const at=t*WIDTH+DIM,s=sequence[at],std=statistics[769];
    out[at]=g[768]/8+(std>0?g[769]*(s-statistics[768])/(8*std):0)+
      (s===statistics[770]?g[770]/minCount:0)+(s===statistics[771]?g[771]/maxCount:0);
  }
  return out;
}
export async function loadHeads(config,base='/model/'){
  const little=new Uint8Array(new Float64Array([1]).buffer)[7]===63;
  return Promise.all(config.members.map(async spec=>{
    const response=await fetch(base+spec.file);if(!response.ok)throw Error('영상 가중치를 불러오지 못했습니다.');
    const buffer=await response.arrayBuffer();
    const hash=[...new Uint8Array(await crypto.subtle.digest('SHA-256',buffer))].map(b=>b.toString(16).padStart(2,'0')).join('');
    if(hash!==spec.sha256)throw Error('영상 가중치 무결성 검사가 실패했습니다.');
    const member={gamma:spec.gamma};
    for(const [name,array]of Object.entries(spec.arrays)){
      if(little)member[name]=new Float64Array(buffer,array.offset,array.length);
      else{const view=new DataView(buffer,array.offset,array.length*8);member[name]=Float64Array.from({length:array.length},(_,i)=>view.getFloat64(i*8,true));}
    }
    return member;
  }));
}
