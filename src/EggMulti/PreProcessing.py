import theano
import numpy as np
import os
import random

class PreProcessing(object):
    nSample = 500
    #nInStep = 50
    #nInput = 3
    #nOutStep = 1
    #nOutput = 2
    def __init__(self, instep, nin, outstep, nout, readtag=0):
        self.read_tag = readtag
        self.nInput = nin
        self.nInStep = instep
        self.nOutStep = outstep
        self.nOutput = nout
        self.data_x =np.zeros((self.nSample, self.nInStep, self.nInput), dtype=theano.config.floatX)
        self.data_y = np.zeros((self.nSample, self.nOutStep, self.nOutput), dtype=theano.config.floatX)
        self.initdata()
    def initdata(self):
        self.setdata()
    def readfiles(self, sfilepath, lable):
        sample_x = []
        sample_y =[]
        for sfile in os.listdir(sfilepath):
            f = open(sfilepath + sfile)
            sample = []
            for value in f:
                sample.append(float(value))
            sample_x.append(sample)
            sample_y.append(lable)
        return sample_x, sample_y
    def setdata(self):
        samples_x = []
        samples_y = []
        filepath = "../../data/"
        tempx, tempy = self.readfiles(filepath+"F/", 0)         
        samples_x = samples_x + tempx
        samples_y = samples_y + tempy 
        tempx, tempy = self.readfiles(filepath+"N/", 1)         
        samples_x = samples_x + tempx
        samples_y = samples_y + tempy       
        tempx, tempy = self.readfiles(filepath+"O/", 2)         
        samples_x = samples_x + tempx
        samples_y = samples_y + tempy   
        tempx, tempy = self.readfiles(filepath+"S/", 3)         
        samples_x = samples_x + tempx
        samples_y = samples_y + tempy   
        tempx, tempy = self.readfiles(filepath+"Z/", 4)         
        samples_x = samples_x + tempx
        samples_y = samples_y + tempy   
        randarray = np.arange(self.nSample)
        if os.path.exists("randarray.txt"):
            randarray = np.loadtxt("randarray.txt", dtype=np.int32)
            print randarray
        else:
            np.random.shuffle(randarray)
            np.savetxt("randarray.txt", randarray, fmt='%d')
        for i in range(self.nSample):
            r = randarray[i]
            for k in range(self.nInput):
                for j in range(self.nInStep):
                    if self.read_tag ==0:
                        self.data_x[i][j][k] = samples_x[r][j+k*self.nInStep]
                    else:
                        self.data_x[i][j][k] = samples_x[r][k+j*self.nInput]
            self.data_y[i][0][samples_y[r]] = 1
        #print randarray[0]
        #print self.data_x[0]
        #print self.data_y[0]
    def GetData(self):
        return self.data_x, self.data_y
    def GetTrain(self):
        return self.data_x[0:360], self.data_y[0:360]
    def GetTest(self):
        return self.data_x[360:400], self.data_y[360:400]
if __name__ == "__main__":
    data = PreProcessing(50, 5, 1, 5, 1)
    traindata, terget = data.GetTrain()
    print traindata[-1]
    print terget[-1]

    
    
    
    
    
