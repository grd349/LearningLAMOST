import sys
from PyQt5 import QtCore
from PyQt5.QtWidgets import QDialog, QApplication, QPushButton, QVBoxLayout, QShortcut

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import numpy as np

from readFitsSlim import Spectra

class Window(QDialog):
    def __init__(self, parent=None):
        super(Window, self).__init__(parent)
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('Spectral Classification')
        self.setWindowFlags(
        QtCore.Qt.Window |
        QtCore.Qt.CustomizeWindowHint |
        QtCore.Qt.WindowTitleHint |
        QtCore.Qt.WindowCloseButtonHint |
        QtCore.Qt.WindowStaysOnTopHint)       
        
        self.classification = []
        self.spec = Spectra('/data2/cpb405/DR1/*.fits')
        self.spec.specList = self.spec.specList[:5]
        self.index = 0
     
        self.figure = plt.figure()

        self.canvas = FigureCanvas(self.figure)

        self.starButton = QPushButton('Star')
        self.starButton.setStyleSheet("background-color: rgb(31, 119, 180);")
        self.starButton.clicked.connect(self.STAR)
        QShortcut(QtCore.Qt.Key_1, self.starButton, self.starButton.animateClick)
        
        self.galaxyButton = QPushButton('Galaxy')
        self.galaxyButton.setStyleSheet("background-color: rgb(31, 119, 180);")
        self.galaxyButton.clicked.connect(self.GALAXY)
        QShortcut(QtCore.Qt.Key_2, self.galaxyButton, self.galaxyButton.animateClick)
        
        self.unknownButton = QPushButton('Unknown')
        self.unknownButton.setStyleSheet("background-color: rgb(31, 119, 180);")
        self.unknownButton.clicked.connect(self.UNKNOWN)
        QShortcut(QtCore.Qt.Key_3, self.unknownButton, self.unknownButton.animateClick)

        # set the layout
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        layout.addWidget(self.starButton)
        layout.addWidget(self.galaxyButton)
        layout.addWidget(self.unknownButton)
        self.setLayout(layout)
        
        self.plot()

    def plot(self):

        self.figure.clear()

        ax = self.figure.add_subplot(111)
       
        ax.plot(self.spec.specList[self.index].wavelength,self.spec.specList[self.index].flux)
        ax.set_xlabel('Wavelength [Angstroms]')
        ax.set_ylabel('Flux')
        ax.set_yscale('log')
        
        if self.index < (len(self.spec.specList)-1):
            self.index += 1
        else:
            print(self.classification)
            np.savetxt("spectralTrainingSet.csv", self.classification, delimiter=",", fmt="%s")
            self.close()
        # refresh canvas
        self.canvas.draw()
        
    def STAR(self):
        self.classification.append('STAR')
        self.plot()
        
    def GALAXY(self):
        self.classification.append('GALAXY')
        self.plot()
        
    def UNKNOWN(self):
        self.classification.append('UNKNOWN')
        self.plot()
        print(self.classification)

if __name__ == '__main__':
    app = QApplication(sys.argv)

    main = Window()
    main.show()
    
    sys.exit(app.exec_())
