"""Reuse the actual deployed, float16-quantized image classifier in TensorFlow."""
import json
from pathlib import Path
import numpy as np
from video_pipeline import ROOT, file_sha256
from baseline_common import weight_hash  # Sets bounded CPU and project cache defaults.
import tensorflow as tf
from google.protobuf.json_format import ParseDict

SOURCE=ROOT.parent/'web/model'

def source_identity():
    manifest=json.loads((SOURCE/'model.json').read_text(encoding='utf-8'))
    files=['model.json','inference_config.json']+[p for group in manifest['weightsManifest'] for p in group['paths']]
    return {p:file_sha256(SOURCE/p) for p in files}

def load_image_backbone(include_gradcam=False):
    manifest=json.loads((SOURCE/'model.json').read_text(encoding='utf-8'))
    graph=tf.compat.v1.GraphDef()
    ParseDict(manifest['modelTopology'],graph)
    nodes={node.name:node for node in graph.node}
    for group in manifest['weightsManifest']:
        content=b''.join((SOURCE/p).read_bytes() for p in group['paths']);offset=0
        for item in group['weights']:
            dtype=item.get('quantization',{}).get('dtype',item['dtype'])
            if dtype not in ('float16','float32','int32','bool'):raise ValueError('Unsupported weight storage: '+dtype)
            count=int(np.prod(item['shape'],dtype=np.int64));size=count*np.dtype(dtype).itemsize
            array=np.frombuffer(content,dtype=np.dtype(dtype).newbyteorder('<'),count=count,offset=offset).reshape(item['shape']);offset+=size
            array=array.astype(item['dtype'])
            node=nodes[item['name']]
            if node.op!='Const':raise ValueError('Expected constant weight')
            node.attr['value'].tensor.CopyFrom(tf.make_tensor_proto(array))
        if offset!=len(content):raise ValueError('Weight shard byte count mismatch')
    if include_gradcam:
        # TF.js fused dense kernels lack a registered TensorFlow gradient.
        # Expand only these kernels to equivalent differentiable operations;
        # prediction continues to use the untouched frozen SavedModel.
        expanded=tf.compat.v1.GraphDef()
        for node in graph.node:
            if node.op!='_FusedMatMul':
                expanded.node.add().CopyFrom(node)
                continue
            ops=list(node.attr['fused_ops'].list.s)
            if ops not in ([b'BiasAdd'],[b'BiasAdd',b'Sigmoid']):
                raise ValueError('Unsupported fused dense operation for Grad-CAM')
            mat=expanded.node.add();mat.name=node.name+'/gradcam_matmul';mat.op='MatMul'
            mat.input.extend(node.input[:2]);mat.attr['T'].CopyFrom(node.attr['T'])
            for attr in ('transpose_a','transpose_b'):mat.attr[attr].CopyFrom(node.attr[attr])
            bias=expanded.node.add();bias.name=node.name if len(ops)==1 else node.name+'/gradcam_bias'
            bias.op='BiasAdd';bias.input.extend([mat.name,node.input[2]])
            bias.attr['T'].CopyFrom(node.attr['T'])
            if len(ops)>1:
                activation=expanded.node.add();activation.name=node.name;activation.op='Sigmoid'
                activation.input.append(bias.name);activation.attr['T'].CopyFrom(node.attr['T'])
        graph=expanded
    features=[n.name for n in graph.node if n.name.endswith('/relu_1/Relu')]
    if len(features)!=1:raise ValueError('Ambiguous trained image embedding')
    wrapped=tf.compat.v1.wrap_function(lambda:tf.compat.v1.import_graph_def(graph,name=''),[])
    input_name=manifest['signature']['inputs']['image']['name']
    output_name=manifest['signature']['outputs']['probability_real']['name']
    function=wrapped.prune(wrapped.graph.get_tensor_by_name(input_name),
                           [wrapped.graph.get_tensor_by_name(features[0]+':0'),wrapped.graph.get_tensor_by_name(output_name)])
    if function.outputs[0].shape[-1]!=256:raise ValueError('Unexpected trained image feature size')
    info={'source_files_sha256':source_identity(),'embedding_tensor':features[0]+':0',
                     'score_tensor':output_name,'embedding_size':256,'storage':'existing TF.js float16 weights expanded to float32; original full precision not available',
                     'image_output':'P(real), uncalibrated score','video_output':'P(fake), uncalibrated score',
                     'input_size':256,'input_range':[0,255],'source_preprocessing':'EfficientNetV2S built-in preprocessing; no /255'}
    if include_gradcam:
        spatial=[n.name for n in graph.node if '/top_activation' in n.name and n.op=='Mul']
        if len(spatial)!=1:raise ValueError('Ambiguous final spatial activation')
        activation=wrapped.graph.get_tensor_by_name(spatial[0]+':0')
        if activation.shape.rank!=4:raise ValueError('Grad-CAM requires spatial activations')
        prefix=wrapped.prune(wrapped.graph.get_tensor_by_name(input_name),activation)
        suffix=wrapped.prune(activation,[wrapped.graph.get_tensor_by_name(features[0]+':0'),
                                        wrapped.graph.get_tensor_by_name(output_name)])
        info['gradcam_tensor']=activation.name
        return function,info,prefix,suffix
    return function,info

if __name__=='__main__':
    tf.config.threading.set_intra_op_parallelism_threads(4);tf.config.threading.set_inter_op_parallelism_threads(1)
    model,info=load_image_backbone()
    f,p=model(tf.zeros((1,256,256,3),tf.float32))
    print(json.dumps({'features':f.shape.as_list(),'real_score':p.numpy().tolist(),'source':info},indent=2),flush=True)
