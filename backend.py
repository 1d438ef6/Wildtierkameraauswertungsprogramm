import pathlib
from PyQt5.QtCore import QThread, pyqtSignal, QObject
import pandas as pd
import cv2
import numpy as np
import os
import pytesseract
import torch
import matplotlib
import matplotlib.pyplot as plt
from deskew import determine_skew
import easyocr
from typing import Tuple, Union
import math
import time

from string_merger import Tree





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
    #img_to_yuv = cv2.cvtColor(image,cv2.COLOR_BGR2YUV)
    #img_to_yuv[:,:,0] = cv2.equalizeHist(img_to_yuv[:,:,0])
    #return cv2.cvtColor(img_to_yuv, cv2.COLOR_YUV2BGR)
    return cv2.equalizeHist(image)

def detecting_boxes(directory,saveTo,settings:dict={}):
    t1 = time.time()

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
    c = ['Kamera','Bild','Bild_Nr','nr','Datum','Zeit','Box_n','Klasse','Box','LP','Wahrscheinlichkeit']
    vertexter = Vertexter(settings['tess'],settings['ocr'])
    nof = 0
    for i in os.listdir(f"{directory}\\"):
        for j in os.listdir(f"{directory}\\{i}\\"):
            nof = nof + len(os.listdir(f"{directory}\\{i}\\{j}\\"))
    print('Number of Pictures in Folders: ', nof)
    counter = 0
    for i in os.listdir(f"{directory}\\"):
        for j in os.listdir(f"{directory}\\{i}\\"):
            for k in os.listdir(f"{directory}\\{i}\\{j}\\"):
                image = cv2.imread(f"{directory}\\{i}\\{j}\\{k}",0)
                orig_img = image
                if settings['improvements']['hist_eq']: image = hist_eq(image)
                if settings['improvements']['neg']: image = negativ(image)
                if settings['improvements']['gamma']: image = gamma_correction(image)
                if settings['improvements']['clahe']: image = clahe(image)
                pred = model(image)
                date_time = get_date_time(vertexter.get_text_from_image_2(orig_img[0:20,0:330]))   # gibt 2 elementige list zurück [0] == Datum, [1] == zeit
                #print(date_time)
                c_2 = 0
                if len(pred.xyxy[0])>0:
                    for b in pred.xyxy[0]:          # pred.xyxy ist ein Tensor mit den Detektionen, aufgebaut sind diese [X1, Y1, X2, Y2, Wahrscheinlichkeit, Klasse]
                        # v steht für Vektor und ist nur eine temporäre Variable
                        # glücklich Georg?
                        v = [ a[0] if a[0] in i else a[1],
                            f"{directory}/{i}/{j}/{k}",
                            counter,
                            vertexter.get_text_from_image_2(orig_img[0:25,425:500])[0][1],
                            date_time[0],
                            date_time[1],
                            c_2,
                            class_names[int(b[-1])],
                            b[0:4],
                            vertexter.get_text_from_lp(orig_img[int(b[1]):int(b[3]),int(b[0]):int(b[2])]) if class_names[int(b[-1])] == 'Nummernschild' else '',
                            b[-2]
                            ]
                        detection.append(v)
                        c_2 = c_2 + 1
                        del v
                else: 
                    detection.append([
                        a[0] if a[0] in i else a[1],
                        f"{directory}/{i}/{j}/{k}",
                        counter,
                        vertexter.get_text_from_image_2(orig_img[0:25,425:500])[0][1],
                        date_time[0],
                        date_time[1],
                        0,
                        "?",
                        [0,0,0,0],
                        '',
                        0])
                print(f"Aktuell: {counter}/{nof}")
                counter = counter + 1
    data = pd.DataFrame(detection,columns=c)
    
    save_csv = False
    if settings['save_csv']: data.to_csv(F"{''.join(saveTo.split('.')[:-1])}.csv")
    
    for cam in a:
        df = data[data['Kamera']==cam]
        ifd = 1
        h = []
        for i in set(df['Bild_Nr']):
            df_temp = df[df['Bild_Nr']==i]
            #print(df_temp['Bild'])
            #print('--------------------------------------------------------------')
            Bild = df_temp['Bild'].iloc[0]
            Datum = df_temp['Datum'].iloc[0]
            Zeit = df_temp['Zeit'].iloc[0]
            nr = df_temp['nr'].iloc[0]
            det_obj = []
            lp = []
            for j in range(len(df_temp)):
                det_obj.append((df_temp['Klasse'].iloc[j],"{:.2f}".format(float(df_temp['Wahrscheinlichkeit'].iloc[j])*100)))
                lp.append(df_temp['LP'].iloc[j])
            h.append([ifd, nr, [k for k in lp if not k == ''], str(det_obj).replace('\'','').strip('[]'), Datum, Zeit, f"=HYPERLINK(\"{Bild}\",\"{Bild}\")"])
            ifd += 1
        if cam == 'Einfahrt':
            Einfahrt = pd.DataFrame(h,columns=['ifd-Nr','Nr','Nummernschilder','Detektierte Objekte','Datum','Zeit','Pfad'])
        else:     
            Ausfahrt = pd.DataFrame(h,columns=['ifd-Nr','Nr','Nummernschilder','Detektierte Objekte','Datum','Zeit','Pfad'])
           

    with pd.ExcelWriter(saveTo) as writer:
        Einfahrt.to_excel(writer, sheet_name="Einfahrt",index=False)
        Ausfahrt.to_excel(writer, sheet_name="Ausfahrt",index=False)

    print('################################################################')
    print('Needed Time: ',time.time()-t1)
            

def get_date_time(timestring)->list():
    #print(timestring)
    try:
    #    tmp = timestring.split(' ')
    #    #mp2 = tmp[0 if len(tmp[0])>2 else 1].split('-')
    #    return [tmp[0 if len(tmp[0])>2 else 1], tmp[1 if len(tmp[0])>2 else 2]]
        return [timestring[0][1], timestring[1][1].replace('*', ':')]
    except:
        return [f"00-00-0000", f"00:00:0000"]



class Vertexter:
    def __init__(self, path_to_tesseract:str="", settings:dict={}):
        self.tesseract = settings['tesseract']
        self.easyocr = settings['easyOCR']
        self.fast = settings['fast']
        self.slow = settings['slow']
        if settings['tesseract']:
            pytesseract.pytesseract.tesseract_cmd = path_to_tesseract
            self.path = path_to_tesseract
        
        self.reader = easyocr.Reader(['en']) # this needs to run only once to load the model into memory

    def rotate(self,image: np.ndarray, angle: float, background: Union[int, Tuple[int, int, int]]) -> np.ndarray:
        old_width, old_height = image.shape[:2]
        angle_radian = math.radians(angle)
        width = abs(np.sin(angle_radian) * old_height) + abs(np.cos(angle_radian) * old_width)
        height = abs(np.sin(angle_radian) * old_width) + abs(np.cos(angle_radian) * old_height)

        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        rot_mat[1, 2] += (width - old_width) / 2
        rot_mat[0, 2] += (height - old_height) / 2
        return cv2.warpAffine(image, rot_mat, (int(round(height)), int(round(width))), borderValue=background)
    
    def get_text_from_lp(self, image)->str:
        orig_img = image
        if self.slow:
            img = cv2.resize(orig_img, (orig_img.shape[1]*7, orig_img.shape[0]*7), interpolation=cv2.INTER_LANCZOS4)
            angle = determine_skew(img)
            if(angle):
                desk_img = self.rotate(img, angle, (0,0,0))
            else: desk_img = img
        if self.fast:
            img1 = cv2.resize(orig_img, (orig_img.shape[1]*10, orig_img.shape[0]*10), interpolation=cv2.INTER_AREA)
            if not self.slow: angle = determine_skew(img1)
            if(angle):
                res_img = self.rotate(img1, angle, (0,0,0))
            else: res_img = img1

        text = []

        if self.fast:
            if self.tesseract: 
                t1 = pytesseract.image_to_string(res_img, lang='eng',config='-c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 7 --oem 3')
                text.append(t1)
            if self.easyocr: 
                t2 = self.reader.readtext(res_img)
                text.append(''.join([j[1] for j in t2]) if len(t2)>0 else '')
        if self.slow:
            if self.tesseract: 
                t3 = pytesseract.image_to_string(desk_img, lang='eng',config='-c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 7 --oem 3')
                text.append(t3)
            if self.easyocr:
                t4 = self.reader.readtext(desk_img)
                text.append(''.join([j[1] for j in t4]) if len(t4)>0 else '')
        
        if len(text)>0:
            l = max([len(i) for i in text])
            tree = Tree(word=text[0], n_layer = l)
            for i in range(1,len(text)):
                tree.add(word=text[i])

            return ''.join([layer.avg() for layer in tree.layers])
        else: return ""

    def get_text_from_image_2(self, image):
        #return pytesseract.image_to_string(image, lang='eng',config='--psm 7 --oem 3')
        return self.reader.readtext(image)

if __name__ == "__main__":
    print("Test OCR")
    vertexter = Vertexter("D:\\Programme\\tesseract\\tesseract.exe")
    # 208, 212
    i = "G:\\Autobilder_Dataset_lp\\308.jpg"
    image = cv2.imread(i)
    print(vertexter.get_text_from_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
    reader = easyocr.Reader(['en']) # this needs to run only once to load the model into memory
    result = reader.readtext(i)
    print(result)
    plt.imshow(image)
    plt.xticks([])
    plt.yticks([])
    plt.show()