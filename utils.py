import math
import cv2
import numpy as np
from insightface.model_zoo.landmark import Landmark
from insightface.model_zoo.retinaface import RetinaFace
from insightface.app.common import Face


class Face_Detection():
    def __init__(self, model_face='det_10g.onnx', input_size = (640,640)):
        self.input_size = input_size
        self.app = RetinaFace(model_file=model_face)
    def get(self, img):
        faces, _ = self.app.detect(img, max_num=0, metric='default', input_size=self.input_size)
        box = faces.astype(np.int)[0]
        return box

