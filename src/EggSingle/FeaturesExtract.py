#coding:utf-8
import numpy as np
from numpy import array
from theano.scalar.basic import sgn

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

if __name__ == "__main__":
    fe = FeaturesExtract()
    temp = np.arange(5)
    print np.sum(abs(temp[1:]) - abs(temp[:-1]))
    print temp.shape[0]
    pass



    

    
