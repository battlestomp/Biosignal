from sklearn.neural_network import MLPClassifier


def readdata(filename):
    fp = open(filename)
    pe =[]
    for line in fp:
        xi = []
        strlist = line.split(",")
        for j in range(len(strlist)):
            xi.append(float(strlist[j]) )
        pe+= [xi]
    return pe

def setlabel(setlen, lable):
    sy = []
    for i in range(setlen): 
        sy+=[lable]  
    return sy

def gettraindata(pe, l, v):
    sx=[]
    sy=[]
    for i in range(len(pe)):
        if i<l:
            sy+= [v]
            sx+= [pe[i]]
    return sx, sy


def gettestdata(pe, l, v):
    tx=[]
    ty=[]
    count = 0;
    for i in range(len(pe)-1, 0, -1):
        if count<l:
            ty+= [v]
            tx+= [pe[i]]
            count = count + 1
        else:
            break
    return tx, ty


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

FA = readdata("../../data/SetAfeastrures") 
FB = readdata("../../data/SetBfeastrures") 
FC = readdata("../../data/SetCfeastrures") 
FD = readdata("../../data/SetDfeastrures") 
FE = readdata("../../data/SetEfeastrures") 
sax, say = gettraindata(FA, 80, 1)
sbx, sby = gettraindata(FB, 80, 2)
scx, scy = gettraindata(FC, 80, 3)
sdx, sdy = gettraindata(FD, 80, 4)
sex, sey = gettraindata(FE, 80, 5)
sx = sax + sbx + scx + sdx + sex
sy = say + sby + scy + sdy + sey
clf = MLPClassifier(algorithm='l-bfgs', alpha=1e-5, hidden_layer_sizes=(8, 5), random_state=1)
clf.fit(sx, sy)

tax, tay = gettestdata(FA, 20, 1)
tbx, tby = gettestdata(FB, 20, 2)
tcx, tcy = gettestdata(FC, 20, 3)
tdx, tdy = gettestdata(FD, 20, 4)
tex, tey = gettestdata(FE, 20, 5)
tx = tax + tbx + tcx + tdx + tex
ty = tay + tby + tcy + tdy + tey


py=clf.predict(tx)
print clf.score(tx, ty)
