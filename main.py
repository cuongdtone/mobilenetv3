
import cv2
import numpy as np
from utils import Face_Detection


face_detect = Face_Detection()
input_path = '/home/cuong/Desktop/Speaking/test/Speaking/video_1_2.mp4'
path_save_video = 'video.mp4'
cap = cv2.VideoCapture(input_path)

while True:
    ret, image = cap.read()
    if ret:
        box = face_detect.get(image)
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0,0,255), 1)
        print(box)
        cv2.imshow('img', image)
        cv2.waitKey(5)
    else:
        break
        print('het')
plt.show()