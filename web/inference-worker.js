// Keep TF parsing, graph creation, shader compilation and inference off the UI thread.
let resources, suffix, library, queue = Promise.resolve();
const diagnosing=new URLSearchParams(location.search).has('diagnostics');
const metric = (name,data={}) => {if(diagnosing||name==='model-progress')postMessage({metric:name,data:{...data,workerAt:performance.timeOrigin+performance.now()}});};
async function ready() {
  if (resources) return resources;
  resources = (async () => {
    if (!library) { importScripts('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js'); library = true; }
    await tf.ready();
    const {MODEL_BASE} = await import('./model-release.js');
    const response = await fetch(MODEL_BASE+'inference_config.json');
    if (!response.ok) throw Error('모델 설정을 불러오지 못했어요.');
    const config = await response.json(); let model;
    try {
      model = await tf.loadGraphModel(MODEL_BASE+'model.json', {onProgress:p=>metric('model-progress',{progress:p})});
      // Yield before compiling. data() waits for actual GPU completion; always dispose.
      await new Promise(resolve=>setTimeout(resolve,0));
      let input, output;
      const before = tf.memory().numTensors;
      try {
        input = tf.zeros([1,...config.image_size,3]);
        metric('warm-predict-start'); output = model.predict(input); metric('warm-predict-end');
        await output.data(); metric('warm-read-end');
      } finally { tf.dispose([input,output].filter(Boolean)); }
      metric('warm-tensors',{before,after:tf.memory().numTensors});
      return {model,config,base:MODEL_BASE};
    } catch(error) { model?.dispose(); throw error; }
  })().catch(error=>{resources=null;throw error;});
  return resources;
}
async function videoReady() {
  const r = await ready();
  if (!suffix) suffix = (async()=>{
    const response=await fetch(r.base+'browser.json'); if(!response.ok)throw Error('영상 설정을 불러오지 못했어요.');
    const config=await response.json(); const model=await tf.loadGraphModel(r.base+config.suffix_model);
    return {model,config};
  })().catch(error=>{suffix=null;throw error;});
  return {...r,video:await suffix};
}
async function run(type,payload) {
  const r=await ready();
  if(type==='init')return {config:r.config,backend:tf.getBackend()};
  if(type==='video-init'){const v=await videoReady();return {config:v.video.config,backend:tf.getBackend()};}
  if(type==='video-gradient') {
    const v=await videoReady(); let activation,gradient;
    try {
      activation=tf.tensor4d(payload.activation,[1,8,8,1280]);
      const weights=payload.weights;
      gradient=tf.tidy(()=>tf.grad(a=>{
        const [embedding,real]=v.video.model.execute(a,[v.video.config.embedding,v.video.config.real]);
        return tf.sum(tf.mul(embedding,tf.tensor2d(weights.slice(0,256),[1,256]))).sub(tf.mul(real,weights[256]).sum());
      })(activation));
      return new Float32Array(await gradient.data());
    } finally {tf.dispose([activation,gradient].filter(Boolean));}
  }
  let input,output;
  try {
    input=tf.tensor4d(payload.rgb,[1,...r.config.image_size,3],'float32');
    if(type==='video-features') {
      const v=await videoReady(); output=r.model.execute(input,[v.video.config.spatial,v.video.config.embedding,v.video.config.real]);
      return await Promise.all(output.map(async tensor=>new Float32Array(await tensor.data())));
    }
    if(type==='explain') {
      const {computeGradcam}=await import('./gradcam.js');return await computeGradcam(tf,r.model,input,payload.real);
    }
    if(type!=='predict')throw Error('지원하지 않는 분석 요청입니다.');
    metric('image-predict-start');output=r.model.predict(input);metric('image-predict-end');
    const raw=(await output.data())[0];metric('image-output',{raw,threshold:r.config.decision_threshold_real});return raw;
  } finally {tf.dispose([input,...(Array.isArray(output)?output:[output])].filter(Boolean));}
}
onmessage=({data:{id,type,payload}})=>{
  queue=queue.then(async()=>{
    try{postMessage({id,value:await run(type,payload)});}catch(error){postMessage({id,error:error.message});}
    // Network timing from the worker (main document timings do not contain these).
    if(diagnosing)metric('worker-requests',{requests:performance.getEntriesByType('resource').map(e=>({url:e.name,start:e.startTime,ms:e.duration,bytes:e.transferSize,body:e.decodedBodySize}))});
  });
};
