#coding:utf-8
import numpy as np
from pylab import *
from FeaturesExtract import * 
from matplotlib.pyplot import savefig

def FFTfig():
    sampling = 512.0
    fftsize = 10240.0
    frestep = sampling/fftsize
    array1_y = np.loadtxt("../../data/Data_F_50/Data_F_Ind0045.txt", delimiter=',', usecols=(0,), dtype=float)
    array2_y = np.loadtxt("../../data/Data_N_50/Data_N_Ind0045.txt", delimiter=',', usecols=(1,), dtype=float)
    freqs = np.linspace(0, sampling/2, fftsize/2+1)
    sp1 = np.abs(np.fft.rfft(array1_y)/fftsize)
    sp2 = np.abs(np.fft.rfft(array2_y)/fftsize)
    print np.mean(sp1[:20/frestep])
    print np.mean(sp2[:200])
    features = FeaturesExtract()
    res = features.GetFFTbyStep(array1_y, [10/frestep, 25/frestep, 50/frestep, 100/frestep, ])
    print res
    subplot(211)
    plot(freqs[:200], sp1[:200])
    subplot(212)
    plot(freqs, sp2)
    show()

def FFTfig(id):
    sampling = 173.6
    fftsize = 4097.0
    frestep = sampling/fftsize
    close('all')
    array1_y = np.loadtxt("../../data/N/N%03d.txt"%id, delimiter=',', usecols=(0,), dtype=float)
    array2_y = np.loadtxt("../../data/Z/Z%03d.txt"%id, delimiter=',', usecols=(0,), dtype=float)
    array3_y = np.loadtxt("../../data/O/O%03d.txt"%id, delimiter=',', usecols=(0,), dtype=float)
    array4_y = np.loadtxt("../../data/S/S%03d.txt"%id, delimiter=',', usecols=(0,), dtype=float)
    array5_y = np.loadtxt("../../data/F/F%03d.txt"%id, delimiter=',', usecols=(0,), dtype=float)
    freqs = np.linspace(0, sampling/2, fftsize/2+1)
    sp1 = np.abs(np.fft.rfft(array1_y)/fftsize)
    sp2 = np.abs(np.fft.rfft(array2_y)/fftsize)
    sp3 = np.abs(np.fft.rfft(array3_y)/fftsize)
    sp4 = np.abs(np.fft.rfft(array4_y)/fftsize)
    sp5 = np.abs(np.fft.rfft(array5_y)/fftsize)
    subplot(511)
    plot(sp1[1:])
    subplot(512)
    plot(sp2[1:])
    subplot(513)
    plot(sp3[1:])
    subplot(514)
    plot( sp4[1:])
    subplot(515)
    plot(sp5[1:])
    # [[310.2934037339271, 4.351858448838163], [154.51730996796798, 186.25194176649248]]
    print np.mean(sp2[10.2934037339271:4.351858448838163])
    #return np.max(sp2[1:])
    #strname = "E:/wentingc/paper/jxst/2/pic/%03d.png"%id
    #savefig(strname, dpi=200)
    #show()
      
    
def fig(id):
    sampling = 173.6
    fftsize = 4097.0
    frestep = sampling/fftsize
    close('all')
    array1_y = np.loadtxt("../../data/N/N%03d.txt"%id, delimiter=',', usecols=(0,), dtype=float)
    array2_y = np.loadtxt("../../data/Z/Z%03d.txt"%id, delimiter=',', usecols=(0,), dtype=float)
    array3_y = np.loadtxt("../../data/O/O%03d.txt"%id, delimiter=',', usecols=(0,), dtype=float)
    array4_y = np.loadtxt("../../data/S/S%03d.txt"%id, delimiter=',', usecols=(0,), dtype=float)
    array5_y = np.loadtxt("../../data/F/F%03d.txt"%id, delimiter=',', usecols=(0,), dtype=float)
    subplot(511)
    plot(array1_y)
    subplot(512)
    plot(array2_y)
    subplot(513)
    plot(array3_y)
    subplot(514)
    plot(array4_y)
    subplot(515)
    plot(array5_y)
    #return np.max(sp2[1:])
    strname = "E:/wentingc/paper/jxst/2/pic2/%03d.png"%id
    savefig(strname, dpi=200)
    #show()
      

def main():
    array1_y = np.loadtxt("../../data/Data_N_50/Data_N_Ind0012.txt", delimiter=',', usecols=(0,), dtype=float)
    array2_y = np.loadtxt("../../data/Data_N_50/Data_N_Ind0012.txt", delimiter=',', usecols=(1,), dtype=float)
    
    array3_y = np.loadtxt("../../data/Data_N_50/Data_N_Ind0012.txt", delimiter=',', usecols=(0,), dtype=float)
    array4_y = np.loadtxt("../../data/Data_N_50/Data_N_Ind0012.txt", delimiter=',', usecols=(1,), dtype=float)  
    
    array5_y = np.loadtxt("../../data/Data_F_50/Data_F_Ind0023.txt", delimiter=',', usecols=(0, 1), dtype=float, unpack=True)  
    
    subplot(211)
    plot(array1_y - array2_y)
    subplot(212)
    plot(array3_y + array4_y)

    show()


if __name__ == "__main__":
    #for i in range(1,100):
    FFTfig(1)
    



    

    
