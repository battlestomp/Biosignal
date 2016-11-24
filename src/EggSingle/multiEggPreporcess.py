#!/usr/bin/env python
# encoding: utf-8
import numpy as np
import os
from FeaturesExtract import *
from numpy import linalg as la
from pyevolve import GSimpleGA
from pyevolve import G1DList
from pyevolve import Selectors
from pyevolve import Initializators, Mutators
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

def orgindatax(sfilepath):
    for (root, dirs, files) in os.walk(sfilepath):
        pass
    
    data_x = []
    for sfilename in files:
        data_x.append(np.loadtxt(sfilepath + sfilename, delimiter = ',', usecols = (0,), dtype = float))
    
    return data_x

data_F_x = orgindatax('../../data/F/')
data_N_x = orgindatax('../../data/N/')
data_O_x = orgindatax('../../data/O/')
data_S_x = orgindatax('../../data/S/')
data_Z_x = orgindatax('../../data/Z/')
g_acc = 0

def getfeaturse(data_x, steplist):
    features = FeaturesExtract()
    continue
    rlist = [ features.GetFFTbyStep(x, steplist) for x in data_x ]
    return np.array(rlist)


def GetXY(data_x, steplist, label_y):
    x = getfeaturse(data_x, steplist)
    y = np.ones((x.shape[0], 1)) * label_y
    return np.concatenate((x, y), axis = 1)


def LoadFeatures(F_x, N_x, O_x, S_x, Z_x, ratio, steplist):
    icount = 0
    pF = GetXY(F_x, steplist, 1)
    pN = GetXY(N_x, steplist, 2)
    pO = GetXY(O_x, steplist, 3)
    pS = GetXY(S_x, steplist, 4)
    pZ = GetXY(Z_x, steplist, 5)
    train_data = np.concatenate((pF[:ratio], pN[:ratio], pO[:ratio], pS[:ratio], pZ[:ratio]), axis = 0)
    test_data = np.concatenate((pF[ratio:], pN[ratio:], pO[ratio:], pS[ratio:], pZ[ratio:]), axis = 0)
    np.random.shuffle(train_data)
    sx = train_data[:, :-1]
    sy = train_data[:, -1]
    tx = test_data[:, :-1]
    ty = test_data[:, -1]
    return (sx, sy, tx, ty)


def eval_func(chromosome):
    steplist = []
    lastvalue = 0
    for index in range(0, 12, 2):
        temp1 = chromosome[index]
        temp2 = chromosome[index + 1]
        if temp1 == temp2:
            temp2 = temp2 + 5
        templist = [
            temp1,
            temp2]
        templist.sort()
        steplist.append(templist)
    
    (sx, sy, tx, ty) = LoadFeatures(data_F_x, data_N_x, data_O_x, data_S_x, data_Z_x, 80, steplist)
    clf = MLPClassifier()
    clf.fit(sx, sy)
    py = clf.predict(tx)
    sorce = accuracy_score(ty, py)
    print sorce, steplist
    return sorce


def ga():
    genome = G1DList.G1DList(12)
    genome.setParams(rangemin = 1, rangemax = 2000)
    genome.initializator.set(Initializators.G1DListInitializatorReal)
    genome.mutator.set(Mutators.G1DListMutatorRealGaussian)
    genome.evaluator.set(eval_func)
    ga = GSimpleGA.GSimpleGA(genome)
    ga.selector.set(Selectors.GRouletteWheel)
    ga.setGenerations(100)
    ga.evolve(freq_stats = 5)
    best = ga.bestIndividual()
    print best


def test():
    steplist = [
        [
            0,
            30],
        [
            1179,
            1180]]
    pF = GetXY(data_F_x, steplist, 1)
    pN = GetXY(data_N_x, steplist, 2)
    pO = GetXY(data_O_x, steplist, 3)
    pS = GetXY(data_S_x, steplist, 4)
    pZ = GetXY(data_Z_x, steplist, 5)
    import matplotlib.pyplot as plt
    (fig, axes) = plt.subplots(nrows = 1, ncols = 2)
    fatures0 = [ pF[:, 0], pN[:, 0], pO[:, 0],pS[:, 0], pZ[:, 0]]
    axes[0].boxplot(fatures0)
    axes[0].set_title('$f_1$', fontsize = 20)
    axes[1].boxplot([pF[:, 1], pN[:, 1], pO[:, 1], pS[:, 1], pZ[:, 1]])
    axes[1].set_title('$f_2$', fontsize = 20)
    plt.show()

if __name__ == '__main__':
    ga()
