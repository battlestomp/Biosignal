import numpy as np

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
