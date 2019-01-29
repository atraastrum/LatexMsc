from mxnet import nd, autograd
import training_data

# NDArray contains matrix of feature vectors from training set.
x = nd.array(training_data.X)
# NDArray contains vector of labels from the training set
# corresponding to feature vectors from x.
l = nd.array(training_data.y)
# NDArray for synaptic weights.
w = nd.array([[1], [1]])
# Allocating memory for gradient
w.attach_grad()
# NDArray for bias
b = nd.array([1])
# Allocating memory for gradient
b.attach_grad()


learning_rate = 0.1
for epoch in range(100):
    # Recording graph
    with autograd.record():
        v = nd.dot(x, w) + b
        loss = nd.log(1 + nd.exp(-v * l))
        emr = nd.mean(loss)
    # Calculating gradient
    emr.backward()
    w[:] = w - learning_rate * w.grad
    b[:] = b - learning_rate * b.grad

print("Synaptic Weights")
print("w_1 = {:.3}, w_2 = {:.3}".format(w[0,0].asscalar(), w[1,0].asscalar()))
print("Bias")
print("b = {:.3}".format(b.asscalar()))

