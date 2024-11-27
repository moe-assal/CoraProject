from PyQt5 import QtWidgets, uic
from PyQt5.QtGui import QIcon

from PyQt5.QtWidgets import *



app = QtWidgets.QApplication([])
dlg= uic.loadUi(r"C:\\Users\\mersh\\OneDrive\\Desktop\\CoraProject\\CORA-UI\\CORA.ui")


dlg.lineEdit.setPlaceholderText("Enter Paper Name")


dlg.show()
app.exec()