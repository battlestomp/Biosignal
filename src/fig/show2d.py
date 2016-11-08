# -*- coding: utf-8 -*-  
'''
Created on 2015/04/23

@author: pake
'''

import matplotlib.pyplot as plt
from copy import deepcopy
from cProfile import label
import numpy as np
from matplotlib.lines import Line2D


def ReadDataFile(filename):
    fp = open(filename, 'r')              
    DataList = []
    try:
        for line in fp.readlines():
            DataList.append(line.strip().split(" "))
    finally:
        fp.close()
    return DataList

if __name__ == '__main__':
    Datalist = ReadDataFile("tempfig")
    print Datalist[0][9], Datalist[0][11]

    markers = []
    for m in Line2D.markers:
        try:
            if len(m) == 1 and m != ' ':
                markers.append(m)
        except TypeError:
            pass
    styles = markers + [r'$\lambda$',r'$\bowtie$',r'$\circlearrowleft$', r'$\clubsuit$', r'$\checkmark$']
    xlist = []
    ylist1 = []
    ylist2 = []
    icount = 0
    for line in Datalist:
        xlist.append(icount)
        ylist1.append(line[9])
        ylist2.append(line[11])
        icount = icount + 1

    plt.grid(True)    

    plt.title('GAPA3', fontsize=22)
    plt.xlabel(r"$F_c(x)$", fontsize=20)
    plt.ylabel(r'$F_b(x)$' , fontsize=20)
    style = styles[11]
    #plot1 = plt.plot(xlist, ylist1,  linestyle='None', marker=style, color='b', markersize=10)
    plot2 = plt.plot(xlist, ylist2,  linestyle='None', marker=style, color='r', markersize=10)

    plt.legend( loc = 'upper left', shadow=True)
    plt.show()
    
    