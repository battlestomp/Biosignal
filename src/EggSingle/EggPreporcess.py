#coding:utf-8
import numpy as np
import os
from FeaturesExtract import * 
from numpy import linalg as la  
from pyevolve import GSimpleGA
from pyevolve import G1DList
from pyevolve import Selectors
from pyevolve import Initializators, Mutators
from sklearn.lda import LDA
from sklearn.metrics import accuracy_score

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

def orgindatax(sfilepath):
    for root, dirs ,files in os.walk(sfilepath):
        pass
    data_x = []
    for sfilename in files:
        data_x.append(np.loadtxt(sfilepath + sfilename, delimiter=',', usecols=(0,), dtype=float))
    return data_x

data_N_x = orgindatax("../../data/Data_N_50/")
data_F_x = orgindatax("../../data/Data_F_50/")


def getfeaturse(data_x, steplist):
    features = FeaturesExtract()
    rlist = [features.GetFFTbyStep(x, steplist) for x in data_x]
    return np.array(rlist)
    

def LoadFeatures(N_x, F_x, chromosome):
    steplist = []
    icount = 0
    for value in chromosome: 
        steplist.append(int((value+icount*51)*20))   
        icount = icount + 1
    pe_x = getfeaturse(F_x, steplist) 
    ne_x = getfeaturse(N_x, steplist)
    pe_y = np.ones((pe_x.shape[0],1))
    ne_y = np.zeros((ne_x.shape[0],1))
    pe = np.concatenate((pe_x, pe_y), axis=1)
    ne = np.concatenate((ne_x, ne_y),axis=1)
    all_data = np.concatenate((pe, ne), axis=0)
    np.random.shuffle(all_data)
    return all_data

def GetData(ratio, all_data):
    all_len = all_data.shape[0]
    s_len = all_len*ratio
    p_y = all_data[:, -1]
    p_x = all_data[:, :-1]
    sx = p_x[:s_len]
    sy = p_y[:s_len]
    tx = p_x[s_len:]
    ty = p_y[s_len:]
    return sx, sy, tx, ty

def eval_func(chromosome):
    alldata = LoadFeatures(data_N_x, data_F_x, chromosome)
    sx, sy, tx, ty = GetData(0.8, alldata)
    clf = LDA()
    clf.fit(sx, sy)
    py = clf.predict(tx)
    return accuracy_score(ty, py)

def ga():
    genome = G1DList.G1DList(5)
    genome.setParams(rangemin=1, rangemax=51)
    genome.initializator.set(Initializators.G1DListInitializatorReal)
    genome.mutator.set(Mutators.G1DListMutatorRealGaussian)
    genome.evaluator.set(eval_func)
    ga = GSimpleGA.GSimpleGA(genome)
    ga.selector.set(Selectors.GRouletteWheel)
    ga.setGenerations(2)
    ga.evolve(freq_stats=10)
    best = ga.bestIndividual()
    print best



def main():
    #eggfeatures("../../data/Data_N_50/")
    #eggfeatures("../../data/Data_F_50/")
    GetCox("../../data/Data_N_50/")
    GetCox("../../data/Data_F_50/")

    
if __name__ == "__main__":
     ga()



    

    
