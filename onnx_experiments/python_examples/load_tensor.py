import onnx
from onnx import numpy_helper

itensor = onnx.TensorProto()
with open('input_0.pb', 'rb') as f:
    itensor.ParseFromString(f.read())

npitensor = numpy_helper.to_array(itensor)

otensor = onnx.TensorProto()
with open('output_0.pb', 'rb') as f:
    otensor.ParseFromString(f.read())

npotensor = numpy_helper.to_array(otensor)

pass

