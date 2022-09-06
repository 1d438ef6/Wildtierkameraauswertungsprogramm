import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import QThread, pyqtSignal
import pathlib
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMessageBox, QMainWindow, QAction
import json
import os
from threading import *

#from backend import *
#from backend import test_funk

class App(QMainWindow):

    def __init__(self):
        super().__init__()
        self.title = 'Wildtierkameraauswertungsprogramm'
        self.left = 100
        self.top = 100
        self.width = 460
        self.height = 280
        self.initUI()
        self.t1 = Thread()
    
    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        l1 = QLabel(self)
        l1.setText("Quellordnereingabe")
        l1.move(25,2)

        self.openDirectoyText = QLineEdit(self)
        self.openDirectoyText.setFixedWidth(280)
        self.openDirectoyText.move(25,25)
        
        openDirectoryBtn = QPushButton(self)
        openDirectoryBtn.setText("Ordner auswählen")
        openDirectoryBtn.setCheckable(False)
        openDirectoryBtn.clicked.connect(self.openDirectoryDialog)
        openDirectoryBtn.setFixedWidth(125)
        openDirectoryBtn.move(320,25)

        l2 = QLabel(self)
        l2.setText("Speichern nach")
        l2.move(25, 52)

        self.saveToText = QLineEdit(self)
        self.saveToText.setFixedWidth(280)
        self.saveToText.move(25,75)

        saveToBtn = QPushButton(self)
        saveToBtn.setText("Speicherort auswählen")
        saveToBtn.setCheckable(False)
        saveToBtn.clicked.connect(self.saveFileDialog)
        saveToBtn.setFixedWidth(125)
        saveToBtn.move(320,75)

        self.settings = settings()

        settingsBtn = QPushButton(self)
        settingsBtn.setText("Einstellungen")
        settingsBtn.setCheckable(False)
        settingsBtn.move(25,130)
        settingsBtn.clicked.connect(self.settings.show)

        startBtn = QPushButton(self)
        startBtn.setText("START")
        startBtn.setFixedSize(400,100)
        startBtn.setCheckable(False)
        startBtn.move(25,175)
        startBtn.clicked.connect(self.start)

        
        
        self.show()

    def showErrorDialog(self, Error:str=""):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setText("Error")
        msg.setInformativeText(Error)
        msg.setWindowTitle("Error")
        msg.exec_()
    
    def openDirectoryDialog(self):
        dialog = QFileDialog(self, windowTitle='Select directory')
        dialog.setFileMode(dialog.Directory)
        dialog.setOptions(dialog.DontUseNativeDialog)
        directoryName = dialog.getExistingDirectory(self,"Ordner wählen", "")
        if directoryName:
            print(directoryName)
            #if check_if_directory_is_valid(directory=pathlib.Path(directoryName)):
            self.openDirectoyText.setText(directoryName)
            #else:
            #    self.showErrorDialog(f"Das ausgewählte Verzeichnis {directoryName} ist nicht valid")


    def openFileNameDialog(self):
        dialog = QFileDialog(self, windowTitle='Select directory')
        dialog.setOptions(dialog.DontUseNativeDialog)
        fileName = dialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "")
        if fileName:
            print(fileName)
    
    def saveFileDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getSaveFileName(self,"speichern nach","","All Files (*);;Text Files (*.txt)", options=options)
        if fileName:
            print(fileName)
            self.saveToText.setText(fileName)
    
    def start(self):
        txt = self.openDirectoyText.text()
        d = self.load_settings()
        print(d)
        self.t1=Thread(target=test_funk,args=(txt,"",d),daemon=True)
        self.t1.start()
        

    def load_settings(self):
        if os.path.exists('settings.json'):
            with open('settings.json','r') as json_file:
                my_dict = json.load(json_file)
            return my_dict

    

    #def closeEvent(self, event):
    #    print("stop")
    #    if self.t1.is_alive:
    #        self.t1.stop()
    #    event.accept()
    

class settings(QtWidgets.QWidget):
    def __init__(self):
        super(settings, self).__init__()
        self.resize(400, 300)
        self.title = 'Einstellungen'
        self.setWindowTitle(self.title)
        self.initUI()
         
    def initUI(self):
        # Label
        l1 = QLabel(self)
        l1.setText("Modeleingabe")
        l1.move(25,2)
        self.modelText = QLineEdit(self)
        self.modelText.setFixedWidth(230)
        self.modelText.move(25,25)
        modelBtn = QPushButton(self)
        modelBtn.setText('Model wählen')
        modelBtn.move(260,25)
        modelBtn.setCheckable(False)
        modelBtn.clicked.connect(self.openModelNameDialog)

        l2 = QLabel(self)
        l2.setText("Pfad zur OCR Anwendung")
        l2.move(25,52)
        self.ocrText = QLineEdit(self)
        self.ocrText.setFixedWidth(230)
        self.ocrText.move(25,70)
        ocrBtn = QPushButton(self)
        ocrBtn.setText('Anwendung wählen')
        ocrBtn.move(260,70)
        ocrBtn.setCheckable(False)
        ocrBtn.clicked.connect(self.openOCRNameDialog)

        l3 = QLabel(self)
        l3.setText("Bildverbesserungen")
        l3.move(25,110)
        self.neg = QCheckBox(self)
        self.neg.setText('Negativ')
        self.neg.move(25,130)

        self.gamma = QCheckBox(self)
        self.gamma.setText('Gamma Korrektion')
        self.gamma.move(25,150)

        self.clahe = QCheckBox(self)
        self.clahe.setText('CLAHE')
        self.clahe.move(160,130)

        self.hist_eq = QCheckBox(self)
        self.hist_eq.setText('Histogram Equalization')
        self.hist_eq.move(160,150)

        okBtn = QPushButton(self)
        okBtn.setText('OK')
        okBtn.setFixedWidth(100)
        okBtn.move(150,250)
        okBtn.setCheckable(False)
        okBtn.clicked.connect(self.save_settings)

        self.load_settings()

    def openModelNameDialog(self):
        dialog = QFileDialog(self, windowTitle='Select directory')
        dialog.setOptions(dialog.DontUseNativeDialog)
        fileName = dialog.getOpenFileName(self,"Modelauswahl", self.modelText.text(), "")
        if fileName:
            print(fileName[0])
            self.modelText.setText(fileName[0])
    
    def openOCRNameDialog(self):
        dialog = QFileDialog(self, windowTitle='Select directory')
        dialog.setOptions(dialog.DontUseNativeDialog)
        fileName = dialog.getOpenFileName(self,"OCR Auswahl", self.ocrText.text(),"")
        if fileName:
            self.ocrText.setText(fileName[0])

    def save_settings(self):
        my_dict = {
            'model' : self.modelText.text(),
            'ocr' : self.ocrText.text(),
            'improvements' : {
                'neg': self.neg.checkState() == 2,
                'gamma': self.gamma.checkState() == 2,
                'clahe': self.clahe.checkState() == 2,
                'hist_eq': self.hist_eq.checkState() == 2
            }
        }
        with open('settings.json', 'w', encoding ='utf8') as json_file:
            json.dump(my_dict, json_file)
        self.close()

    def load_settings(self):
        if os.path.exists('settings.json'):
            with open('settings.json','r') as json_file:
                my_dict = json.load(json_file)
            self.modelText.setText(my_dict['model'])
            self.ocrText.setText(my_dict['ocr'])
            i = my_dict['improvements']
            if i['neg']: self.neg.setChecked(True)
            if i['gamma']: self.gamma.setChecked(True)
            if i['clahe']: self.clahe.setChecked(True)
            if i['hist_eq']: self.hist_eq.setChecked(True)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
