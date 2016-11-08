from PreProcessing import PreProcessing
from PIL import Image
from numpy import *
import os

class Data2Img():
    def __init__(self, data):
        self.data = data
        self.pic_w = data[0].shape[0]
        self.pic_h = data[0].shape[1]    
        self.maxpixel = data.max()
        self.minpixel = data.min()
    def calcpix(self, pix):
        value = (pix-self.minpixel)/self.maxpixel * 255
        return value
    def image_create(self, picdata, sfile):
        im = Image.new("L", (self.pic_w, self.pic_h), (0))
        for i in range(self.pic_w):
            for j in range(self.pic_h):
                value = self.calcpix(picdata[i][j])
                im.putpixel((i, j), value)
        im.save(sfile)
    def work(self):
        for i in range(self.data.shape[0]):
            filepath = "../../data/pic/"
            if i<100:
                sfile = filepath + "F" + str(i) + ".png"
            elif i>=100 and i<200:
                sfile = filepath + "N" + str(i-100) + ".png"
            elif i>=200 and i<300:
                sfile = filepath + "O" + str(i-200) + ".png"
            elif i>=300 and i<400:
                sfile = filepath + "S" + str(i-300) + ".png"
            elif i>=400 and i<500:
                sfile = filepath + "Z" + str(i-400) + ".png"
            self.image_create(self.data[0], sfile)

if __name__ == "__main__":
    data = PreProcessing(60, 60, 1, 5, 1)
    traindata, terget = data.GetData()
    data2img = Data2Img(traindata)
    data2img.work()
