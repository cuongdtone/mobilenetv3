import cv2
import numpy as np
from utils import Face_Detection, Face_Mask, draw
from glob import glob
from mobilenet_v3.pred_mobilenetv3 import mobilenetv3

net = mobilenetv3(weights='model/classifier.h5')
path = '/home/cuong/Desktop/Face mask/dataset/val/with_mask/'
list_img = glob(path+'/*.png')
for i in list_img:
    image = cv2.imread(i)
    pred = net.predict(image)

    cv2.imshow('%d'%(pred), image)
    cv2.waitKey()
