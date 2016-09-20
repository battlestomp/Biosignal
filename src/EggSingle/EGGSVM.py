#coding:utf-8
'''
Created on 2016锟斤拷6锟斤拷16锟斤拷

@author: pake
'''

import numpy as np
from numpy import float32, float16
from sklearn import svm

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
        if i<50:
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
    setA = readdata("../data/SetAfeastrures") 
    syA = setlabel(len(setA), 1.0)
    setB = readdata("../data/SetEfeastrures") 
    syB = setlabel(len(setB), -1.0)
    return syA+syB, setA + setB

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

if __name__ == '__main__':
    pe = readdata("../data/SetAfeastrures") 
    ne = readdata("../data/SetEfeastrures") 
    sx, sy, tx, ty = randomdata(pe, ne)
    ry, rx = readtraindata()
    m = svm_train(ry, rx, "-s 0  -c 2 -g 0.3")
    p_label, p_acc, p_val = svm_predict(ty, tx, m)
    svm_save_model("svm.txt", m)
    print evaluations(ty, p_label);
    print SenAndSpe(ty, p_label);
    pass
