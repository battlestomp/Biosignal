#coding:utf-8
'''
Created on 2016锟斤拷6锟斤拷16锟斤拷

@author: pake
'''

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

from pyevolve import G1DBinaryString
from pyevolve import GSimpleGA
from pyevolve import Selectors
from pyevolve import Mutators
from sklearn.lda import LDA


def GetData(ratio):
    all_data = LoadFeatures()
    all_len = all_data.shape[0]
    s_len = all_len*ratio
    p_y = all_data[:, -1]
    p_x = all_data[:, :-1]
    
#    pca=PCA(30)
#    p_x = pca.fit_transform(p_x)

    sx = p_x[:s_len]
    sy = p_y[:s_len]
    tx = p_x[s_len:]
    ty = p_y[s_len:]
    return sx, sy, tx, ty
def LoadFeatures():
    file_pe = "../../data/features/data_F_result1EMDfft.txt"
    file_ne = "../../data/features/data_N_result1EMDfft.txt"
    pe_x = np.loadtxt(fname=file_pe, dtype=float, delimiter=",", unpack=False)
    ne_x = np.loadtxt(fname=file_ne, dtype=float, delimiter=",", unpack=False)
    pe_y = np.ones((pe_x.shape[0],1))
    ne_y = np.zeros((ne_x.shape[0],1))
    pe = np.concatenate((pe_x, pe_y), axis=1)
    ne = np.concatenate((ne_x, ne_y),axis=1)
    all_data = np.concatenate((pe, ne), axis=0)
    np.random.shuffle(all_data)
    return all_data


def classify(sx, sy, tx, ty):
    clf = LDA()
    clf.fit(sx, sy)
    py = clf.predict(tx)
    return accuracy_score(ty, py)

sx, sy, tx, ty = GetData(0.8)
def resetx(sx, tx, subarray):
    csx = np.delete(sx, subarray, 1)
    ctx = np.delete(tx, subarray, 1)
    return csx, ctx
    
def eval_func(chromosome):
    subarray = []
    icout = 0
    for value in chromosome:
        if value == 0:
            subarray.append(icout)
        icout = icout + 1
    csx, ctx = resetx(sx, tx, subarray)
    sorce = classify(csx, sy, ctx, ty)
    print sorce
    return sorce
def ga():
    genome = G1DBinaryString.G1DBinaryString(sx.shape[1])
    genome.evaluator.set(eval_func)
    genome.mutator.set(Mutators.G1DBinaryStringMutatorFlip)
    ga = GSimpleGA.GSimpleGA(genome)
    ga.selector.set(Selectors.GTournamentSelector)
    ga.setGenerations(20)
    ga.evolve(freq_stats=10)
    best = ga.bestIndividual()
    print best
    #print eval_func(best)

def multi_classifier():
    classifiers = [
                   KNeighborsClassifier(4),
                   SVC(kernel="linear", C=0.025),
                   SVC(),
                   #####GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
                   #DecisionTreeClassifier(max_depth=7),
                   #RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
                   #RandomForestClassifier(),
                   LDA(),
                   AdaBoostClassifier(),
                   #GaussianNB(),
                   #QuadraticDiscriminantAnalysis()
                   ]
    for clf in classifiers:
        clf.fit(sx, sy)
        py = clf.predict(tx)
        print accuracy_score(ty, py)
if __name__ == '__main__':
    #ga()
    print multi_classifier()
    #print sx.shape
    pass
