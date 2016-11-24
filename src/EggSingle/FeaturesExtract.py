#coding:utf-8
import numpy as np
from numpy import array
from theano.scalar.basic import sgn
import nolds

class FeaturesExtract:
    def __init__(self):
        pass
    def GetIVA(self, farray=array):
        return np.sum(abs(farray))
    def GetMAV(self, farray=array):
        return np.sum(abs(farray))/farray.shape(0)
    def GetRMS(self, farray=array):
        return np.sqrt(sum(farray**2))
    def GetWL(self, farray=array):
        return np.sum(farray[1:]-farray[:-1])
    def GetMAVS(self, farray=array):
        return np.sum(abs(farray[1:]) - abs(farray[:-1]))/(farray.shape[0]-1)
    def GetZC(self, farray=array):
        return 1
    def sgnMatrix(self, farray):
        farray[farray>0] = 1
        farray[farray<0] = 0
    def GetEntropy(self, farry):
        resultlist = []
        resultlist.append(nolds.sampen(farry))
        resultlist.append(nolds.lyap_r(farry))
        resultlist.append(nolds.hurst_rs(farry))
        resultlist.append(nolds.dfa(farry))
    def GetFFTbyStep(self, farry, steparray):
        resultlist = []
        sp = np.abs(np.fft.rfft(farry)/farry.shape[0])
        for step in steparray:
            resultlist.append(np.mean(sp[step[0]:step[1]]))
        return np.array(resultlist)
    def GetFFTMax(self, farry, steparray):
        resultlist = []
        sp = np.abs(np.fft.rfft(farry)/farry.shape[0])
        for step in steparray:
            resultlist.append(np.max(sp[step[0]:step[1]]))
        return np.array(resultlist)
    
if __name__ == "__main__":
    testlist = [1,2,3,4]
    for i in range(0, len(testlist), 2):
         print i
    pass



    

    
