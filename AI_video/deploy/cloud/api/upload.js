const {handleUpload}=require('@vercel/blob/client');
const {list,del}=require('@vercel/blob');
const ORIGIN='https://duhyeon-ai-video.vercel.app';
const pattern=/^inputs\/(\d{13})-[a-f0-9]{32}\.mp4$/;
let cleaned=0;
module.exports=async function(request,response){
  response.setHeader('Access-Control-Allow-Origin',ORIGIN);
  response.setHeader('Access-Control-Allow-Methods','POST,OPTIONS');
  response.setHeader('Access-Control-Allow-Headers','Content-Type');
  response.setHeader('Cache-Control','no-store');
  if(request.method==='OPTIONS')return response.status(204).end();
  if(request.method!=='POST')return response.status(405).end();
  try{
    let body=request.body;
    if(!body){let size=0;const chunks=[];for await(const c of request){size+=c.length;if(size>16384)throw Error('요청이 너무 큽니다.');chunks.push(c);}body=JSON.parse(Buffer.concat(chunks));}
    if(typeof body==='string')body=JSON.parse(body);
    const result=await handleUpload({body,request,
      onBeforeGenerateToken:async pathname=>{
        if(request.headers.origin!==ORIGIN)throw Error('허용되지 않은 요청입니다.');
        const match=pattern.exec(pathname);
        if(!match||Math.abs(Date.now()-Number(match[1]))>120000)throw Error('업로드를 다시 시작해 주세요.');
        if(Date.now()-cleaned>300000){
          cleaned=Date.now();
          try{const page=await list({prefix:'inputs/',limit:100});
            const expired=page.blobs.filter(b=>pattern.test(b.pathname)&&Date.now()-Number(pattern.exec(b.pathname)[1])>3600000).map(b=>b.url);
            if(expired.length)await del(expired);
          }catch{/* Quota/cleanup failure must not publish a private file. */}
        }
        return {maximumSizeInBytes:90*1024*1024,validUntil:Date.now()+300000,
          allowedContentTypes:['video/*','application/octet-stream'],addRandomSuffix:false,allowOverwrite:false};
      },onUploadCompleted:async()=>{}});
    return response.status(200).json(result);
  }catch(error){return response.status(400).json({error:error.message||'업로드 준비에 실패했습니다.'});}
};
