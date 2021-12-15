
import cv2
import numpy as np
from utils import Face_Detection, Face_Mask, draw


face_detect = Face_Mask(model_face='model/det_10g.onnx', model_mask='model/classifier.h5')
input_path = 0#'/home/cuong/Desktop/Speaking/GreenGlobal/Tram_Chinh_Dien.mp4'
path_save_video = 'video.mp4'
cap = cv2.VideoCapture(input_path)

while True:
    ret, image = cap.read()
    if ret:
        boxes, label = face_detect.get(image)
        image = draw(image, boxes, label)
        cv2.imshow('img', image)
        cv2.waitKey(5)
    else:
        break
        print('het')
plt.show()