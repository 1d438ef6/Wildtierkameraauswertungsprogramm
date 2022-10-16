import pathlib
from PyQt5.QtCore import QThread, pyqtSignal, QObject
import pandas as pd
import cv2
import numpy as np
import os

import torch


# prüft ob ausgewähltes Verzeichnis gültig ist
# basiert auf der Annahme dass die Daten immer gleich aufgebaut sind
def check_if_directory_is_valid(directory="")->bool:
    content = list(directory.iterdir())
    if (not ((directory/'Einfahrt') in content and (directory/'Ausfahrt') in content)) and len(content)!=2:
        return False
    if len(list((directory/'Einfahrt').iterdir))!=2 and len(list((directory/'Einfahrt').iterdir))!=2:
        return False
    

    return True

def gamma_correction(image):
    gamma = 2
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")

	# apply gamma correction using the lookup table
    return cv2.LUT(image, table)

def clahe(image):
    clahe = cv2.createCLAHE(clipLimit = 5)
    return clahe.apply(image)

def negativ(image):
    return 1 - image

def hist_eq(image):
    img_to_yuv = cv2.cvtColor(image,cv2.COLOR_BGR2YUV)
    img_to_yuv[:,:,0] = cv2.equalizeHist(img_to_yuv[:,:,0])
    return cv2.cvtColor(img_to_yuv, cv2.COLOR_YUV2BGR)

def detecting_boxes(directory,saveTo,settings:dict={}):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=settings['model'], force_reload=True)
    model.eval()

    class_names = {}
    for n in model.names.keys():
        if model.names[n] == 'license plate':
            class_names[n] = 'Nummernschild'
        else:
            class_names[n] = model.names[n]

    a = ['Einfahrt','Ausfahrt']
    detection = []
    c = ['Kamera','Bild','Datum','Zeit','Box_n','Klasse','Box','LP','Wahrscheinlichkeit']
    vertexter = Vertexter(settings['ocr'])
    for i in os.listdir(f"{directory}\\"):
        for j in os.listdir(f"{directory}\\{i}\\"):
            for k in os.listdir(f"{directory}\\{i}\\{j}\\"):
                image = cv2.imread(f"{directory}\\{i}\\{j}\\{k}")
                if settings['improvements']['neg']: image = negativ(image)
                if settings['improvements']['gamma']: image = gamma_correction(image)
                if settings['improvements']['clahe']: image = clahe(image)
                if settings['improvements']['hist_eq']: image = hist_eq(image)
                pred = model(image)
                date_time = get_date_time(vertexter.get_text_from_image(img[0:20,0:330]))
                c_2 = 0
                for b in pred.xyxy[0]:          # pred.xyxy ist ein Tensor mit den Detektionen, aufgebaut sind diese [X1, Y1, X2, Y2, Wahrscheinlichkeit, Klasse]
                    # v steht für Vektor und ist nur eine temporäre Variable
                    # glücklich Georg?
                    v = [ a[0] if a[0] in i else a[1],
                          k,
                          date_time[0],
                          date_time[1],
                          c_2,
                          class_names[int(b[-1])],
                          b[0:4],
                          vertexter.get_text_from_image(img[b[0]:b[2],b[1]:b[3]]) if class_names[int(b[-1])] == 'Nummernschild' else '',
                          b[-2]
                        ]
                    detection.append(v)
                    c_2 = c_2 + 1
                    del v
    data = pd.DataFrame(detection,columns=c)
    data.to_csv(saveTo)
            
            

def get_date_time(timestring)->list():
    pass



class Vertexter:
    def __init__(self, path_to_tesseract:str=""):
        self.pytesseract.pytesseract.tesseract_cmd = path_to_tesseract
        self.path = path_to_tesseract
    
    def get_text_from_image(image)->str:
        return self.pytesseract.image_to_string(image, lang='eng')


