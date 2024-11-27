import sys
import cv2
from PyQt5 import QtWidgets
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QDialog, QApplication
from PyQt5.uic import loadUi
from PyQt5.QtGui import QImage, QPixmap
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.preprocessing.image import img_to_array
import urllib.request
import numpy as np
from PIL import Image,ImageChops

class predictGUI(QDialog):
    def __init__(self):
        super(predictGUI,self).__init__()
        loadUi(r"C:\Users\Rzl\Documents\tugas s2\hijaiyah.ui",self)
        self.predictButton.clicked.connect(self.predict_frame)
        self.model = models.load_model(r"C:\Users\Rzl\Documents\tugas s2\Hijaiyah-Deployment-CNN\models\model_0.885_1.000.h5")
        self.timer = QTimer(self)
        self.video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 531)
        self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 411)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(20)
        self.fig = None
        self.pred = None
        self.label_mapping = {
            0: 'Ain',
            1: 'Alif',
            2: 'Ba',
            3: 'dal',
            4: 'dhod',
            5: 'dzal',
            6: 'dzho',
            7: 'fa',
            8: 'ghoin',
            9: 'ha',
            10: 'ha',
            11: 'hamzah',
            12: 'jim',
            13: 'kaf',
            14: 'kho',
            15: 'lam',
            16: 'lamalif',
            17: 'mim',
            18: 'nun',
            19: 'qof',
            20: 'ro',
            21: 'shod',
            22: 'sin',
            23: 'syin',
            24: 'ta',
            25: 'tho',
            26: 'tsa',
            27: 'wawu',
            28: 'ya',
            29: 'zaiAinn'
        }
    
    def update_frame(self):
        ret, frame = self.video_capture.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            q_img = QImage(frame, w, h, bytes_per_line, QImage.Format_RGB888)
            self.videoLive.setPixmap(QPixmap.fromImage(q_img))
            self.fig = cv2.resize(frame,(150,150),interpolation = cv2.INTER_LINEAR)

    def predict_frame(self):
        self.fig = img_to_array(self.fig)
        self.fig = self.cleaned_image(self.fig)
        self.fig = np.expand_dims(self.fig,axis=0)
        self.pred = self.model.predict(self.fig)
        self.pred *= 100
        self.resultLabel.setText(self.get_text_representation())

    def get_text_representation(self):
        max_index = np.argmax(self.pred)
        max_value = self.pred[0][max_index]
        if max_value > 0:
            return self.label_mapping.get(max_index, 'Tidak Diketahui')
        else:
            return 'Tidak Diketahui'
        
    def normalize(self,I):
        mn=I.min()
        mx=I.max()
        mx-=mn
        I=((I-mn)/mx)*255
        return I.astype(np.uint8)

    def trim(self,image):
        #image=Image.fromarray((image* 255).astype(np.uint8))
        image = self.normalize(image)
        image=Image.fromarray(image)
        bg = Image.new(image.mode, image.size, image.getpixel((0,0))) # black background
        diff = ImageChops.difference(image, bg)
        diff = ImageChops.add(diff, diff, 2.0, -100)
        bbox = diff.getbbox()
        if bbox:
            return np.array(image.crop(bbox))
    # cv2.INTER_AREA: examiner les pixels voisins et utiliser ces voisins pour augmenter ou diminuer optiquement la taille de l’image sans introduire de distorsions
    def image_resize(self,image, width = None, height = None, inter = cv2.INTER_AREA):
        dim = None
        (h, w) = image.shape[:2]
        if width is None and height is None:
            return image
        if width is None:
            r = height / float(h)
            dim = (int(w * r), height)
        else:
            #r = width / float(w)
            #dim = (width, int(h * r))
            dim = (width,height)
        resized = cv2.resize(image, dim, interpolation = inter)
        return resized
        
    def cleaned_image(self,image):
        bg=np.ones((150,150))
        bg = bg*255
        #image=self.trim(image)
        image = self.normalize(image)
        image=np.array(image)
        image=self.image_resize(image,height=150)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh, image_binary = cv2.threshold(image_gray, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        hh, ww = bg.shape
        h, w = image_binary.shape
        yoff = round((hh-h)/2)
        xoff = round((ww-w)/2)
        if xoff<=0:
            image=self.image_resize(image,height=150,width=150)
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            thresh, image_binary = cv2.threshold(image_gray, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            h, w = image_binary.shape
            yoff = round((hh-h)/2)
            xoff = round((ww-w)/2)
        result = bg.copy()
        result[yoff:yoff+h, xoff:xoff+w] = image_binary
        #result = np.expand_dims(result,axis=2)
        result = np.stack((result,)*3, axis=-1)
        return result


if __name__ == "__main__":
    app=QApplication(sys.argv)
    mainwindow=predictGUI()
    widget=QtWidgets.QStackedWidget()
    widget.addWidget(mainwindow)
    widget.setFixedWidth(672)
    widget.setFixedHeight(572)
    widget.show()
    app.exec_()