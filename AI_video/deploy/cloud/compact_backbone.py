"""Compile the same frozen graph as a prefix and a small differentiable suffix.

Avoid importing the full convolution graph into the Grad-CAM gradient function.
No weight, input transform, threshold or video-head parameter is modified.
"""
import gc
import json
import numpy as np
import tensorflow as tf
from google.protobuf.json_format import ParseDict


def build(backbone):
    manifest = json.loads((backbone.SOURCE/'model.json').read_text())
    base = tf.compat.v1.GraphDef()
    ParseDict(manifest['modelTopology'], base)
    spatial = [n.name for n in base.node if '/top_activation' in n.name and n.op=='Mul']
    embedding = [n.name for n in base.node if n.name.endswith('/relu_1/Relu')]
    if len(spatial)!=1 or len(embedding)!=1:
        raise ValueError('Ambiguous frozen graph outputs')
    spatial, embedding = spatial[0], embedding[0]
    input_name = manifest['signature']['inputs']['image']['name']
    output_name = manifest['signature']['outputs']['probability_real']['name']
    nodes = {n.name:n for n in base.node}

    def subgraph(outputs, stop=None):
        needed=set()
        def visit(name):
            name=name.lstrip('^').split(':')[0]
            if name in needed: return
            needed.add(name)
            if name != stop:
                for parent in nodes[name].input: visit(parent)
        for output in outputs: visit(output)
        graph=tf.compat.v1.GraphDef()
        for old in base.node:
            if old.name not in needed: continue
            if old.name==stop:
                node=graph.node.add();node.name=stop;node.op='Placeholder'
                node.attr['dtype'].type=tf.float32.as_datatype_enum
                node.attr['shape'].shape.CopyFrom(tf.TensorShape([None,8,8,1280]).as_proto())
            elif old.op=='_FusedMatMul':
                ops=list(old.attr['fused_ops'].list.s)
                if ops not in ([b'BiasAdd'],[b'BiasAdd',b'Sigmoid']):
                    raise ValueError('Unsupported frozen dense operation')
                mat=graph.node.add();mat.name=old.name+'/cloud_matmul';mat.op='MatMul'
                mat.input.extend(old.input[:2]);mat.attr['T'].CopyFrom(old.attr['T'])
                for attr in ('transpose_a','transpose_b'):mat.attr[attr].CopyFrom(old.attr[attr])
                bias=graph.node.add();bias.name=old.name if len(ops)==1 else old.name+'/cloud_bias'
                bias.op='BiasAdd';bias.input.extend([mat.name,old.input[2]])
                bias.attr['T'].CopyFrom(old.attr['T'])
                if len(ops)>1:
                    act=graph.node.add();act.name=old.name;act.op='Sigmoid'
                    act.input.append(bias.name);act.attr['T'].CopyFrom(old.attr['T'])
            else: graph.node.add().CopyFrom(old)
        return graph

    graphs=[subgraph([spatial]), subgraph([embedding,output_name],stop=spatial)]
    for graph in graphs:
        selected={n.name:n for n in graph.node}
        for group in manifest['weightsManifest']:
            content=b''.join((backbone.SOURCE/p).read_bytes() for p in group['paths'])
            offset=0
            for item in group['weights']:
                dtype=item.get('quantization',{}).get('dtype',item['dtype'])
                count=int(np.prod(item['shape'],dtype=np.int64))
                size=count*np.dtype(dtype).itemsize
                if item['name'] in selected:
                    array=np.frombuffer(content,dtype=np.dtype(dtype).newbyteorder('<'),
                                        count=count,offset=offset).reshape(item['shape']).astype(item['dtype'])
                    selected[item['name']].attr['value'].tensor.CopyFrom(tf.make_tensor_proto(array))
                offset+=size
            if offset!=len(content):raise ValueError('Frozen shard size mismatch')
    def compile_graph(graph,input_tensor,shape,outputs):
        def call(value):
            return tf.compat.v1.import_graph_def(graph,name='',input_map={input_tensor:value},
                                                 return_elements=outputs)
        return tf.compat.v1.wrap_function(call,[tf.TensorSpec(shape,tf.float32)])
    prefix_call=compile_graph(graphs[0],input_name,[None,256,256,3],[spatial+':0'])
    suffix=compile_graph(graphs[1],spatial+':0',[None,8,8,1280],[embedding+':0',output_name])
    def prefix(value):return prefix_call(value)[0]
    def encoder(value):return suffix(prefix(value))
    info={'source_files_sha256':backbone.source_identity(),'gradcam_tensor':spatial+':0'}
    del graphs,base,nodes
    gc.collect()
    return encoder,info,prefix,suffix
