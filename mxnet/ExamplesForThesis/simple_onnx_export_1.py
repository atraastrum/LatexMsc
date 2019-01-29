import mxnet as mx
import numpy as np
import onnx
from mxnet import gluon
from mxnet.contrib import onnx as onnx_mxnet

net = gluon.nn.Dense(1, in_units=2)
net.collect_params().initialize()

# collect_params returns gluon.parameter.ParameterDict object
# that contains all updatable parameters of a gluon.Block.
# Keys have type str and values have type mxnet.gluon.parameter.Parameter.
# data returns content of parameter as an ndarray.
params = {k: v.data(ctx=mx.cpu()) for k, v in net.collect_params().items()}
sym = net(mx.sym.var('data'))
file_path = onnx_mxnet.export_model(
    sym,
    params,
    input_shape=[(1, 2)],
    input_type=np.float32,
    onnx_file_path='simple_onnx_export_1.onnx')

print('Model saved to ' + file_path)

model = onnx.load('simple_onnx_export_1.onnx')
print('Textual representation of graph of ONNX model')
print(model.graph)
