#coding=utf-8
import theano
import numpy as np
from theano import tensor as T
from collections import OrderedDict
from _ctypes import sizeof
from numpy import size

class RNN(object):
    '''neural net model '''
    def __init__(self,ni,nh,nc,lr=0.05,batch_size=64,singleout=True,hidden_activation=T.nnet.sigmoid,output_activation=T.nnet.softmax,cost_function='nll'):
        '''
        ni  :: dimension of the input layer
        nh :: dimension of the hidden layer
        nc :: dimension of the output layer(number of classes)
        singleout  :: true or false
        hidden_activation ::T.nnet.sigmoid or T.tanh
        output_activation  :: T.nnet.softmax
        cost_function :: nll or cxe(0,1) or mse(^2)
        '''
        # parameters of the model
        self.ni = ni
        self.nh = nh
        self.nc = nc
        def init_weight(mx,nx):
            #theano.shared(0.2 * np.random.uniform(-1.0, 1.0,(mx, nx)).astype(theano.config.floatX))
            return np.asarray(np.random.uniform(size=(mx, nx), low=-.01, high=.01), dtype=theano.config.floatX)

        self.Wx =  theano.shared(init_weight(self.ni,self.nh), name="Wx")#input layer weight (ni*nh)
        self.Wh =  theano.shared(init_weight(self.nh,self.nh), name="Wh")#hiden layer weight (nh*nh)
        self.Wo =  theano.shared(init_weight(self.nh,self.nc), name="Wo") #output layer weight (nh*nc)
        self.bh = theano.shared(np.zeros(nh, dtype=theano.config.floatX), name="bh")#bia of hiden (nh)
        self.b = theano.shared(np.zeros(nc, dtype=theano.config.floatX), name="b")#bia of output (nc)
        self.h0 = theano.shared(np.zeros(nh, dtype=theano.config.floatX), name="h0")#init hiden state (nh)
        self.params = [self.Wh, self.Wx, self.Wo, self.h0, self.bh, self.b]
        self.activation = output_activation
        self.hactivation = hidden_activation
        x = T.matrix()
        #two classification or multiple
        if singleout:
            y = T.matrix()
        else:
            y = T.tensor3()

        #iterator
        def recurrence(x_t, h_tm1):
            h_t = self.hactivation(T.dot(x_t, self.Wx) + T.dot(h_tm1, self.Wh) + self.bh)  #hl time_t output=f(xin*Wx+h_t-1*Wh+b)
            s_t = T.nnet.softmax(T.dot(h_t, self.Wo) + self.b)  #time_t output
            return h_t, s_t

        [h, s], _= theano.scan(recurrence, sequences=x, outputs_info=[self.h0, None])

        self.output = s[-1]
        #cost function
        cxe = T.mean(T.nnet.binary_crossentropy(self.output,y))
        nll = -T.mean(y*T.log(self.output)+(1.-y)*T.log(1.-self.output))
        mse = T.mean((self.output-y)**2)
        cost = 0
        if cost_function == 'mse':
            cost = mse
        elif cost_function == 'cxe':
            cost = cxe
        else:
            cost = nll
        #learning rate
        self.lr = T.scalar()
        #self.lr = theano.shared(np.cast[dtype](lr), name='lr')
        #gradients
        gradients = T.grad(cost, self.params)
        updates = OrderedDict((p, p-lr*g) for p, g in zip(self.params, gradients))
        # train
        self.train = theano.function( inputs=[x,y],outputs=cost,updates=updates)
        #loss
        self.loss = theano.function(inputs=[x,y],outputs=cost)
        # theano functions
        self.classify = theano.function(inputs=[x], outputs=self.output)
    #save params
    def save(self, folder):   
        for param, name in zip(self.params, self.names):
            np.save(os.path.join(folder, name + '.npy'), param.get_value())

from MLPClassify import MLPClassify
if __name__ == "__main__":
    mlpclassify = MLPClassify()
    samplelen = len(mlpclassify.SX)
    traindata = np.random.randn(samplelen, 6)
    targets = np.zeros((samplelen, 5))
    for i in range(samplelen):
        targets[i][mlpclassify.SY[i]-1] = 1
        for j in range(6):
            traindata[i][j] = mlpclassify.SX[i][j]
    #print traindata[1:2]
    print targets[0], targets[101]
    
    n_hidden = 1
    n_in = 1
    n_out = 5
    classifier = RNN(n_in, n_hidden, n_out)
    #classifier.train([traindata[1], traindata[2]], [targets[1], targets[2]]);
    for i in range(1000):
        print classifier.train([[1],[2], [3], [3]], [targets[0]]); 
        #print classifier.train([[4],[1], [3], [1]], [targets[101]]); 
    print classifier.classify([[1],[2], [3], [3]])
    print classifier.classify([[4],[1], [3], [1], [2]])
 
            