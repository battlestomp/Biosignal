import numpy
import theano
import theano.tensor as TT

r = TT.dscalar("r")
u = TT.vector("u")

k = TT.dscalar("k")

# activation function
def step(x, pre_result):
    print x
    return x + pre_result, x

# the hidden state `h` for the entire sequence, and the output for the
# entrie sequence `y` (first dimension is always time)
[results, k], updates = theano.scan(step, sequences= [u], outputs_info=[r, None])

fun = theano.function(inputs=[u, r], outputs=[results, k], updates=updates)


print fun([1, 2,2,3], 0)