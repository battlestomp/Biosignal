# -*- coding: utf-8 -*-  
'''
Created on 2016/11/9

@author: pake
'''


import matplotlib.pyplot as plt
import numpy as np
from _imaging import font


# generate some random test data
szmax = np.loadtxt("szmax.txt", dtype=float)
zzmax = np.loadtxt("zzmax.txt" , dtype=float)
datamax = [szmax, zzmax]

szrms = np.loadtxt("szrms.txt", dtype=float)
zzrms = np.loadtxt("zzrms.txt" , dtype=float)
datarms = [szrms, zzrms]

szthree = np.loadtxt("szthree.txt", dtype=float)
zzthree = np.loadtxt("zzthree.txt" , dtype=float)
datathree = [szthree, zzthree]

# plot violin plot
fig, axes = plt.subplots(nrows=1, ncols=3)
axes[0].boxplot(datamax)
axes[0].set_title('$f_1$', fontsize=20)
axes[0].set_xticklabels(("$Index$", "$Middle$"), fontsize=15)
axes[1].boxplot(datarms)
axes[1].set_title('$f_2$' , fontsize=20)
axes[1].set_xticklabels(("$Index$", "$Middle$"), fontsize=15)
axes[2].boxplot(datathree)
axes[2].set_title('$f_3$' , fontsize=20)
axes[2].set_xticklabels(("$Index$", "$Middle$"), fontsize=15)

#plt.setp(axes, xticks=[y+1 for y in range(len(datamax))],
#         xticklabels=['Index', 'Middle']), 
plt.show()
