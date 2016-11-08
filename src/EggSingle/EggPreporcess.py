#coding:utf-8
import numpy as np
import os
from FeaturesExtract import * 

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
    for sfilename in files:
        if sfilename == "cof.txt":
            continue
        data_x = np.loadtxt(sfilepath + sfilename, delimiter=',', usecols=(0, 1), dtype=float, unpack=True)
        r = np.corrcoef(data_x)
        listfeatures.append(str(r[0][1]) + "\n")
    fn = open(sfilepath + "cof.txt", "w")
    fn.writelines(listfeatures)
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



    

    
