import cv2
import sys
from PyQt5.QtWidgets import  QWidget, QLabel, QApplication, QPushButton
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from utils import Face_Detection, Face_Mask, draw

face_detect = Face_Mask(model_face='model/det_10g.onnx', model_mask='model/classifier.h5')

class Thread(QThread):
    changePixmap = pyqtSignal(QImage)
    def __init__(self, width, height, cameraID):
        super(Thread, self).__init__()
        self.width = width
        self.height = height
        self.cameraID = cameraID
    def run(self):
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if ret:
                # https://stackoverflow.com/a/55468544/6622587
                boxes, label = face_detect.get(frame)
                rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgbImage = draw(rgbImage, boxes, label)
                h, w, ch = rgbImage.shape
                bytesPerLine = ch * w
                convertToQtFormat = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
                p = convertToQtFormat.scaled(self.width, self.height, Qt.KeepAspectRatio)
                self.changePixmap.emit(p)


class App(QWidget):
    def __init__(self):
        super().__init__()
        self.title = 'Nhận diện khẩu trang'
        self.left = 100
        self.top = 100
        self.width = 1280
        self.height = 720
        self.initUI()

    @pyqtSlot(QImage)
    def setImage(self, image):
        self.label.setPixmap(QPixmap.fromImage(image))

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.resize(1800, 1200)
        # create a label
        self.label = QLabel(self)
        self.label.move(280, 120)
        self.label.resize(self.width, self.height)
        th = Thread(self.width, self.height, 0)
        th.changePixmap.connect(self.setImage)
        th.start()
        self.show()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())