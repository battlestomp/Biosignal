from sklearn.neural_network import MLPClassifier
from numpy import select

class MLPClassify(object):
    # features of A, B, C, D, E
    FA = []
    FB = []
    FC = []
    FD = []
    FE = []    

    # train data
    SX = []
    SY = []
    
    # test data
    TX = []
    TY = []
    def __init__(self):
        filepaht = "../../data/"
        self.FA = self.readdata(filepaht+"SetAfeastrures") 
        self.FB = self.readdata(filepaht+"SetBfeastrures") 
        self.FC = self.readdata(filepaht+"SetCfeastrures") 
        self.FD = self.readdata(filepaht+"SetDfeastrures") 
        self.FE = self.readdata(filepaht+"SetEfeastrures")
        
        #init train data
        ntraindata = 80
        sax, say = self.gettraindata(self.FA, ntraindata, 1)
        sbx, sby = self.gettraindata(self.FB, ntraindata, 2)
        scx, scy = self.gettraindata(self.FC, ntraindata, 3)
        sdx, sdy = self.gettraindata(self.FD, ntraindata, 4)
        sex, sey = self.gettraindata(self.FE, ntraindata, 5)
        self.SX = sax + sbx + scx + sdx + sex
        self.SY = say + sby + scy + sdy + sey
        
        #init test data
        ntestdata = 20
        tax, tay = self.gettestdata(self.FA, ntestdata, 1)
        tbx, tby = self.gettestdata(self.FB, ntestdata, 2)
        tcx, tcy = self.gettestdata(self.FC, ntestdata, 3)
        tdx, tdy = self.gettestdata(self.FD, ntestdata, 4)
        tex, tey = self.gettestdata(self.FE, ntestdata, 5)
        self.TX = tax + tbx + tcx + tdx + tex
        self.TY = tay + tby + tcy + tdy + tey
        
        
    def readdata(self, filename1):
        fp1 = open(filename1)
        pe =[]
        for line1 in fp1:
            xi = []
            strlist1 = line1.split(",")          
            for i in range(len(strlist1)):
                xi.append(float(strlist1[i]) )
            pe+= [xi]            
        return pe 
    
    def selectfeature(self, data, select):
        newdata = []
        for line in data:
            aline = []
            for i in range(len(line)):
                if select[i]>=100:
                    aline.append(line[i])            
            newdata.append(aline) 
        return newdata   
    
    def setlabel(self, setlen, lable):
        sy = []
        for i in range(setlen): 
            sy+=[lable]  
        return sy
    
    def gettraindata(self, pe, l, v):
        sx=[]
        sy=[]
        for i in range(len(pe)):
            if i<l:
                sy+= [v]
                sx+= [pe[i]]
        return sx, sy
    
    
    def gettestdata(self, pe, l, v):
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
    
    
    def SenAndSpe(self, ty, pv):
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
    
    def classiflyByFeastures(self, select):
        newFA = self.selectfeature(self.FA, select)
        newFB = self.selectfeature(self.FB, select)
        newFC = self.selectfeature(self.FC, select)
        newFD = self.selectfeature(self.FD, select)
        newFE = self.selectfeature(self.FE, select)

    def classify(self, select):  
        print(select)
        newFA = self.selectfeature(self.FA, select)
        newFB = self.selectfeature(self.FB, select)
        newFC = self.selectfeature(self.FC, select)
        newFD = self.selectfeature(self.FD, select)
        newFE = self.selectfeature(self.FE, select)
#         newFF = self.selectfeature(self.FF, select)
        
        if(select[6]<100):
            select += 100;
        traindata = select[6]/2
        testdata = 100-traindata
        
        sax, say = self.gettraindata(newFA, traindata, 1)
        sbx, sby = self.gettraindata(newFB, traindata, 2)
        scx, scy = self.gettraindata(newFC, traindata, 3)
        sdx, sdy = self.gettraindata(newFD, traindata, 4)
        sex, sey = self.gettraindata(newFE, traindata, 5)
        sx = sax + sbx + scx + sdx + sex
        sy = say + sby + scy + sdy + sey
        
        
        n=0;
        hide = []
        for j in range(7,10):
            if(select[j]>0):
                hide.append(select[j])
                n=n+1
        
        if(n==0):
            return 0
        if(n==1):    
            clf = MLPClassifier(algorithm='l-bfgs', activation='logistic', alpha=1e-5, 
                            hidden_layer_sizes=(hide[0]), random_state=1)
        if(n==2):    
            clf = MLPClassifier(algorithm='l-bfgs', activation='logistic', alpha=1e-5, 
                            hidden_layer_sizes=(hide[0],hide[1]), random_state=1)
        if(n==3):    
            clf = MLPClassifier(algorithm='l-bfgs', activation='logistic', alpha=1e-5, 
                            hidden_layer_sizes=(hide[0],hide[1],hide[2]), random_state=1)
        clf.fit(sx, sy)
        
        tax, tay = self.gettestdata(newFA, testdata, 1)
        tbx, tby = self.gettestdata(newFB, testdata, 2)
        tcx, tcy = self.gettestdata(newFC, testdata, 3)
        tdx, tdy = self.gettestdata(newFD, testdata, 4)
        tex, tey = self.gettestdata(newFE, testdata, 5)
#         tfx, tfy = self.gettestdata(newFF, 20, 6)

        tx = tax + tbx + tcx + tdx + tex
        ty = tay + tby + tcy + tdy + tey
       
        score = clf.score(tx, ty)
        
        return score
    
if __name__ == "__main__":
    mlpclassify = MLPClassify()
