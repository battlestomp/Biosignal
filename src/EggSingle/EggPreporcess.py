#coding:utf-8
import numpy as np
import os
from FeaturesExtract import * 
from numpy import linalg as la  

#欧式距离  
def euclidSimilar(inA,inB):  
    return 1.0/(1.0+la.norm(inA-inB))  
#皮尔逊相关系数  
def pearsonSimilar(inA,inB):  
    if len(inA)<3:  
        return 1.0  
    return 0.5+0.5*np.corrcoef(inA,inB,rowvar=0)[0][1]  
#余弦相似度  
def cosSimilar(inA,inB):  
    inA=np.mat(inA)  
    inB=np.mat(inB)  
    num=float(inA*inB.T)  
    denom=la.norm(inA)*la.norm(inB)  
    return 0.5+0.5*(num/denom) 

def getfeatures(datax):
    features = FeaturesExtract()
    fx = []
    #fx.append(features.GetRMS(datax))
    #fx.append(features.GetWL(datax))
    #fx.append(features.GetMAVS(datax))
    return fx

def GetCox(sfilepath):
    for root, dirs ,files in os.walk(sfilepath):
        pass
    listfeatures = []
    templist = []
    for sfilename in files:
        
        if "cof.txt" in sfilename:
            continue
        data_x = np.loadtxt(sfilepath + sfilename, delimiter=',', usecols=(0, 1), dtype=float, unpack=True)
        r = np.corrcoef(data_x)
        temp = cosSimilar(data_x[0], data_x[1])
        templist.append(temp)
        listfeatures.append(str(r[0][1]) + "\n")
    #print listfeatures
    print templist
    #fn = open(sfilepath + "cof.txt", "w")
    #fn.writelines(listfeatures)
def FFTfeatures():
    array1_y = np.loadtxt("../../data/Data_F_50/Data_F_Ind0045.txt", delimiter=',', usecols=(0,), dtype=float)
    

def eggfeatures(sfilepath):
    #array1_y = np.loadtxt("../../data/Data_F_50/Data_F_Ind0045.txt", delimiter=',', usecols=(0,), dtype=float)
    #sfilepath = "../../data/Data_N_50/"
    for root, dirs ,files in os.walk(sfilepath):
        pass
    listfeatures = []
    for sfilename in files:
        data_x = np.loadtxt(sfilepath + sfilename, delimiter=',', usecols=(0,), dtype=float)
        listfeatures.append(getfeatures(data_x))
    print listfeatures

def main():
    #eggfeatures("../../data/Data_N_50/")
    #eggfeatures("../../data/Data_F_50/")
    GetCox("../../data/Data_N_50/")
    GetCox("../../data/Data_F_50/")
    
if __name__ == "__main__":
     main()



    

    
