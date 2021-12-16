import numpy as np
import cv2
from insightface.model_zoo.retinaface import RetinaFace
from mobilenet_v3.pred_mobilenetv3 import mobilenetv3


class Face_Detection():
    def __init__(self, model_face='det_10g.onnx', input_size = (640,640)):
        self.input_size = input_size
        self.app = RetinaFace(model_file=model_face)
    def get(self, img):
        faces, _ = self.app.detect(img, max_num=0, metric='default', input_size=self.input_size)
        box = faces.astype(np.int)
        return box
class Face_Mask():
    def __init__(self, model_face='det_10g.onnx', model_mask = ''):
        self.face = Face_Detection(model_face)
        self.classifier = mobilenetv3(model_mask)
    def get(self, image):
        h, w = image.shape[:2]
        boxes = self.face.get(image)
        bboxes = []
        labels = []
        for box in boxes:
            if box[1]>0 and box[3]<h and box[0]>0 and box[2]<w:
                face = image[box[1]: box[3], box[0]:box[2], :]
                #cv2.imshow('face', face)
                #cv2.waitKey(5)
                #classification
                pred = self.classifier.predict(face)
                if pred==1:
                    label = 'Khong deo khau trang'
                elif pred==0:
                    label = 'Co khau trang'
                labels.append(label)
                bboxes.append(box)
        return bboxes, labels

def draw(image, boxes, labels):
    if len(labels)>0:
        for idx, box in enumerate(boxes):
            if labels[idx] == 'Khong deo khau trang':
                color = (255,0,0)
            else:
                color = (0, 0, 255)
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, 2)
            cv2.putText(image,
                        labels[idx], (box[0], box[3]),
                        fontFace=cv2.FONT_HERSHEY_PLAIN,
                        thickness=2,
                        fontScale=1,
                        color=(0,255,255)
                        )
    return image