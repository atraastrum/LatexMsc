import mxnet as mx
import mxnet.ndarray as nd
x = mx.sym.Variable('x')
y = mx.sym.Variable('y')

f = 2*x + 3*y

ex = f.bind(ctx=mx.cpu(),
            args={'x': nd.array([2.]), 'y': nd.array([20.])},
            args_grad={'x': nd.array([2.]), 'y': nd.array([20.])})

ex.forward(is_train=True)

ex.backward(nd.ones(1))

#print(ex.outputs)

print(ex.grad_arrays)

