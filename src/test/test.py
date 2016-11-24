import os
import sys
sys.path.append('D:\\experiment\\workspace\\EGGClassification\\src\\EggSingle')
from FeaturesExtract import * 


def orgindatax(sfilepath):
    for root, dirs ,files in os.walk(sfilepath):
        pass
    data_x = []
    for sfilename in files:
        data_x.append(np.loadtxt(sfilepath + sfilename, delimiter=',', usecols=(0,), dtype=float))
    return data_x

data_F_x = orgindatax("../../data/F/")
data_N_x = orgindatax("../../data/N/")
data_O_x = orgindatax("../../data/O/")
data_S_x = orgindatax("../../data/S/")
data_Z_x = orgindatax("../../data/Z/")

#data_F_x = orgindatax("../../data/Data_F_50/")

def getfeaturse(data_x):
    features = FeaturesExtract()
    rlist = [features.GetEntropy(x) for x in data_x]
    return np.array(rlist)

def main():
    F_x = getfeaturse(data_F_x)
    N_x = getfeaturse(data_N_x)
    O_x = getfeaturse(data_O_x)
    S_x = getfeaturse(data_S_x)
    Z_x = getfeaturse(data_Z_x)
    
    np.savetxt("F_Entropy.out",  F_x, delimiter=',')  
    np.savetxt("N_Entropy.out",  N_x, delimiter=',')  
    np.savetxt("O_Entropy.out",  O_x, delimiter=',')  
    np.savetxt("S_Entropy.out", S_x, delimiter=',')  
    np.savetxt("Z_Entropy.out",  Z_x, delimiter=',')  

    
if __name__ == "__main__":
     main()