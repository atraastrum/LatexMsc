import mxnet as mx
import training_data

# Symbol represents matrix of feature vectors from training set.
x = mx.sym.Variable('x')
# Symbol represents vector of labels from the training set
# corresponding to feature vectors from x.
l = mx.sym.Variable('l')
# Symbol represents synaptic weights of the perceptron.
w = mx.sym.Variable('w')
# Symbol represents bias to the perceptron.
b = mx.sym.Variable('b')

# Induced local field of perceptron.
v = mx.sym.broadcast_add(mx.sym.dot(x, w), b)
# Logistic loss function.
loss =  mx.sym.log(1 + mx.sym.exp(-v * l))
# Empirical Risk.
emr = mx.sym.mean(loss)
# Binding values from training data and initial values for synapses and bias.
ex = emr.bind(ctx=mx.cpu(), args={'x': mx.nd.array(training_data.X),
                                  'l': mx.nd.array(training_data.y),
                                  'w': mx.nd.array([[1], [1]]),
                                  'b': mx.nd.array([1])},
                            args_grad={'b': mx.nd.zeros(1), 'w': mx.nd.zeros((2, 1))})

learning_rate = 0.1
for epoch in range(100):
    # Calculating empirical risk and derivatives.
    ex.forward(is_train=True)
    ex.backward(mx.nd.ones(1))
    ex.arg_dict['w'] -= learning_rate * ex.grad_dict['w']
    ex.arg_dict['b'] -= learning_rate * ex.grad_dict['b']

print("Synaptic Weights")
w_final = ex.arg_dict['w'].asnumpy()
print("w_1 = {:.3}, w_2 = {:.3}".format(w_final[0,0], w_final[1,0]))
print("Bias")
print("b = {:.3}".format(ex.arg_dict['b'].asnumpy()[0]))
