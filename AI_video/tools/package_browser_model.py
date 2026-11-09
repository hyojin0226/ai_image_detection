"""Package unchanged frozen weights for browser inference; no training/evaluation."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request
import numpy as np
import release

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'models/final_research_v1'
OUT=ROOT/'deploy/site/model'

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write_json(path,value):path.write_text(json.dumps(value,separators=(',',':'),ensure_ascii=False),encoding='utf-8')

def main():
    identity=release.check()
    OUT.mkdir(parents=True,exist_ok=True)
    image=SOURCE/'image_model'
    model=json.loads((image/'model.json').read_text())
    for name in ['model.json']+[p for group in model['weightsManifest'] for p in group['paths']]:
        shutil.copyfile(image/name,OUT/name)
    nodes={n['name']:n for n in model['modelTopology']['node']}
    spatial=next(n['name'] for n in nodes.values() if '/top_activation' in n['name'] and n['op']=='Mul')
    embedding=next(n['name'] for n in nodes.values() if n['name'].endswith('/relu_1/Relu'))
    output=model['signature']['outputs']['probability_real']['name']
    needed=set()
    def visit(name):
        name=name.lstrip('^').split(':')[0]
        if name in needed:return
        needed.add(name)
        if name!=spatial:
            for parent in nodes[name].get('input',[]):visit(parent)
    visit(embedding);visit(output)
    selected=[]
    for old in model['modelTopology']['node']:
        if old['name'] not in needed:continue
        node=copy.deepcopy(old)
        if node['name']==spatial:
            node={'name':spatial,'op':'Placeholder','attr':{'dtype':{'type':'DT_FLOAT'},
                'shape':{'shape':{'dim':[{'size':str(n)} for n in [-1,8,8,1280]]}}}}
        elif node['op']=='_FusedMatMul':
            ops=node['attr']['fused_ops']['list']['s']
            # TF.js stores the string list as plain strings in this export.
            if ops not in (['BiasAdd'],['BiasAdd','Sigmoid']):
                import base64
                ops=[base64.b64decode(x).decode() for x in ops]
            if ops not in (['BiasAdd'],['BiasAdd','Sigmoid']):raise ValueError('Unsupported frozen dense kernel')
            matname=node['name']+'/browser_matmul'
            attrs={key:node['attr'][key] for key in ['T','transpose_a','transpose_b']}
            selected.append({'name':matname,'op':'MatMul','input':node['input'][:2],'attr':attrs})
            biasname=node['name'] if len(ops)==1 else node['name']+'/browser_bias'
            selected.append({'name':biasname,'op':'BiasAdd','input':[matname,node['input'][2]],'attr':{'T':node['attr']['T']}})
            if len(ops)>1:selected.append({'name':node['name'],'op':'Sigmoid','input':[biasname],'attr':{'T':node['attr']['T']}})
            continue
        selected.append(node)
    specs=[];chunks=[]
    for group in model['weightsManifest']:
        data=b''.join((image/p).read_bytes() for p in group['paths']);offset=0
        for spec in group['weights']:
            dtype=spec.get('quantization',{}).get('dtype',spec['dtype'])
            length=int(np.prod(spec['shape']))*np.dtype(dtype).itemsize
            if spec['name'] in needed:specs.append(spec);chunks.append(data[offset:offset+length])
            offset+=length
        if offset!=len(data):raise ValueError('Weight byte count mismatch')
    (OUT/'suffix.bin').write_bytes(b''.join(chunks))
    suffix=copy.deepcopy(model);suffix['modelTopology']['node']=selected
    suffix['weightsManifest']=[{'paths':['suffix.bin'],'weights':specs}]
    tensor_shape=lambda shape:{'dim':[{'size':str(n)} for n in shape]}
    suffix['signature']={'inputs':{'activation':{'name':spatial+':0','dtype':'DT_FLOAT','tensorShape':tensor_shape([-1,8,8,1280])}},
      'outputs':{'features':{'name':embedding+':0','dtype':'DT_FLOAT','tensorShape':tensor_shape([-1,256])},
                 'real':{'name':output,'dtype':'DT_FLOAT','tensorShape':tensor_shape([-1,1])}}}
    write_json(OUT/'suffix.json',suffix)
    frozen=json.loads((SOURCE/'frozen.json').read_text());members=[]
    for i,member in enumerate(frozen['selected']['members']):
        arrays=[];metadata={};offset=0
        with np.load(SOURCE/member['file'],allow_pickle=False) as saved:
            for name in ['mean','scale','block','support','coefficient']:
                array=np.asarray(saved[name],dtype='<f8');metadata[name]={'offset':offset,'shape':list(array.shape),'length':array.size}
                arrays.append(array.tobytes());offset+=array.nbytes
        name=f'head-{i+1}.bin';(OUT/name).write_bytes(b''.join(arrays))
        members.append({'file':name,'arrays':metadata,'gamma':member['spec']['gamma'],'sha256':digest(OUT/name)})
    write_json(OUT/'browser.json',{'version':identity['version'],'threshold':.5,'frames':64,'members':members,
        'image_model':'model.json','suffix_model':'suffix.json','spatial':spatial+':0','embedding':embedding+':0','real':output,
        'source_release_sha256':digest(ROOT/'releases/v1.0.0.json'),
        'source_image_sha256':digest(image/'model.json'),'browser_accuracy_validated':False,
        'preprocessing':'Browser video decoding and canvas aspect-preserving RGB resize with black square padding; decoder/resampling can differ from FFmpeg.'})
    vendor=OUT.parent/'vendor';vendor.mkdir(exist_ok=True)
    assets=[('tf.min.js','https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js'),
            ('tfjs-LICENSE.txt','https://www.apache.org/licenses/LICENSE-2.0.txt')]
    for name,url in assets:
        target=vendor/name
        if not target.exists():
            with urllib.request.urlopen(url,timeout=60) as response:target.write_bytes(response.read())
    if digest(vendor/'tf.min.js')!='300dfae273d20b4046f46a06d735688f03675a807561e9bcb5f664eb2f3d2831':
        raise ValueError('Pinned TensorFlow.js library checksum mismatch')
    print('Browser assets prepared; unchanged model weights and threshold. No data/test inference.')

if __name__=='__main__':main()
