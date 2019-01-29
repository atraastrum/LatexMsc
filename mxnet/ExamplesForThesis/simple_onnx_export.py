# Definition of the graph that adds two numbers
import mxnet as mx
import onnx
from mxnet.contrib import onnx as onnx_mxnet


a = mx.sym.var('a')
b = mx.sym.var('b')
net = a + b

file_path = onnx_mxnet.export_model(net, params={}, input_shape=[(1,), (1,)], onnx_file_path='simple_onnx_export.onnx')

print('Model saved to ' + file_path)

model = onnx.load('simple_onnx_export.onnx')

# Type denotation defines semantic type for input or output
# In ONNX documentation it is proposed to have the following standard denotations:
# TENSOR
# IMAGE
# AUDIO
# TEXT
model.graph.input[0].type.denotation = 'TENSOR'
model.graph.input[1].type.denotation = 'TENSOR'
model.graph.output[0].type.denotation = 'TENSOR'

print('Textual representation of graph of ONNX model')
print(model.graph)
onnx.save_model(model, "simple_onnx_export.onnx")



