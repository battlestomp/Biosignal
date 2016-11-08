#coding:utf-8
import numpy as np
from pylab import *

def FFTfig():
    array1_y = np.loadtxt("../../data/Data_F_50/Data_F_Ind0040.txt", delimiter=',', usecols=(0,), dtype=float)
    array2_y = np.loadtxt("../../data/Data_N_50/Data_N_Ind0040.txt", delimiter=',', usecols=(0,), dtype=float)
    array1_x = np.arange(10240.0)/512
    

    freqs = np.linspace(0, 512.0/2, 10240.0/2+1)

    sp1 = abs(np.fft.fft(array1_y))
    sp1[0] = sp1[0]/(10240/2)
    sp1[1:-1] = sp1[1:-1]/(10240.0/2)
 

    sp2 = abs(np.fft.fft(array2_y))
    sp2[0] = sp2[0]/(10240/2)
    sp2[1:-1] = sp2[1:-1]/(10240.0/2)

    #subplot(411)
    #plot(array1_x, array1_y)
    #subplot(412)
    #plot(array1_x, array2_y)
    subplot(211)
    #plot(freqs[2500:], sp1[2500:5121])
    plot(freqs[1250:2500], sp1[1250:2500])
    subplot(212)
    plot(freqs[1250:2500], sp2[1250:2500])
    show()

def main():
    array1_y = np.loadtxt("../../data/Data_N_50/Data_N_Ind0021.txt", delimiter=',', usecols=(0,), dtype=float)
    array2_y = np.loadtxt("../../data/Data_N_50/Data_N_Ind0021.txt", delimiter=',', usecols=(1,), dtype=float)
    
    array3_y = np.loadtxt("../../data/Data_F_50/Data_F_Ind0021.txt", delimiter=',', usecols=(0,), dtype=float)
    array4_y = np.loadtxt("../../data/Data_F_50/Data_F_Ind0021.txt", delimiter=',', usecols=(1,), dtype=float)  
    
    array5_y = np.loadtxt("../../data/Data_F_50/Data_F_Ind0021.txt", delimiter=',', usecols=(0, 1), dtype=float, unpack=True)  
    
    #subplot(211)
    #plot(array1_y + array2_y)
    #subplot(212)
    #plot(array3_y + array4_y)
    
    print np.corrcoef(array1_y, array2_y)
    
    print np.corrcoef(array3_y, array4_y)
    
    r = np.corrcoef(array5_y)
    print r[0][1]
    show()


if __name__ == "__main__":
    main()

    



    

    
