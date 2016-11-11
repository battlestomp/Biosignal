#coding:utf-8
'''
Created on 2016锟斤拷6锟斤拷16锟斤拷

@author: pake
'''

import numpy as np
from numpy import float32, float16
from sklearn import svm

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from svmutil import *


def readdata(filename):
    fp = open(filename)
    pe =[]
    for line in fp:
        xi = {}
        strlist = line.split(",")
        for j in range(len(strlist)):
            xi[j]= float(strlist[j])
        pe+= [xi]
    return pe
def randomdata(pe, ne):
    sx=[]
    sy=[]
    tx=[] 
    ty=[]
    for i in range(len(pe)):
        if i<40:
            sy+= [1.0]
            sx+= [pe[i]]
            sy+= [-1.0]
            sx+= [ne[i]]
        else:
            ty+= [1.0]
            tx+= [pe[i]]
            ty+= [-1.0]
            tx+= [ne[i]]
    return sx, sy, tx, ty   

def setlabel(setlen, lable):
    sy = []
    for i in range(setlen): 
        sy+=[lable]  
    return sy

def readtraindata():
    #setA = readdata("../../data/SetAfeastrures") 
    setA = readdata("../../data/F_features.txt") 
    syA = setlabel(len(setA), 1.0)
    #setB = readdata("../../data/SetEfeastrures") 
    setB = readdata("../../data/N_features.txt") 
    syB = setlabel(len(setB), -1.0)
    return syA+syB, setA + setB
def mergeList(pe1, pe2):
    pe = []
    if len(pe1) != len(pe2):
        raise ValueError("len(ty) must equal to len(pv)")  
    for p1, p2 in zip(pe1, pe2):
        xi = {}
        icount = 0
        for j in range(len(p1)):
            xi[j]= p1[j]
            icount = j 
        icount = icount + 1
        for k in range(len(p2)):
            xi[icount+k] = p2[k]
        pe+= [xi]
    return pe

def SenAndSpe(ty, pv):
    if len(ty) != len(pv):
        raise ValueError("len(ty) must equal to len(pv)")
    total_correct = total_error = 0
    TP = TN = FP = FN = 0
    for v, y in zip(pv, ty):
        if y == v:
            total_correct += 1
            if v == 1:
                TP += 1
            else:
                TN += 1
        else:
            total_error += 1
            if v == 1:
                FP += 1
            else:
                FN += 1
    ACC = (TP+TN)*1.0/(TP+FP+FN+TN)
    SEN = (TP)*1.0/(TP+FN)
    SPC = (TN)*1.0/(TN+FP)
    return SEN, SPC, ACC

def SVMmain():
    pe1 = readdata("../../data/F_features.txt") 
    ne1 = readdata("../../data/N_features.txt") 
    pe2 = readdata("../../data/DataF_features.txt") 
    ne2 = readdata("../../data/DataN_features.txt") 
    pe = mergeList(pe1, pe2)
    ne = mergeList(ne1, ne2)
    sx, sy, tx, ty = randomdata(pe, ne)
    print sx
    #ry, rx = readtraindata()
    m = svm_train(sy,sx, "-s 0  -c 3 -g 0.4")
    p_label, p_acc, p_val = svm_predict(ty, tx, m)
    svm_save_model("svm.txt", m)
    print evaluations(ty, p_label);
    print SenAndSpe(ty, p_label);

def Boostmain():
    pe1 = readdata("../../data/F_features.txt") 
    ne1 = readdata("../../data/N_features.txt") 
    pe2 = readdata("../../data/DataF_features.txt") 
    ne2 = readdata("../../data/DataN_features.txt") 
    pe = mergeList(pe1, pe2)
    ne = mergeList(ne1, ne2)
    sx, sy, tx, ty = randomdata(pe, ne)
    print sx
    #ry, rx = readtraindata()
    bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                         algorithm="SAMME",
                         n_estimators=200)
    bdt.fit(sy, sx)
    print bdt.predict(tx)

if __name__ == '__main__':
    Boostmain()
    
    pass
