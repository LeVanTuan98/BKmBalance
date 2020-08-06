# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'GUI.ui'
#
# Created by: PyQt5 UI code generator 5.12.3
#
# WARNING! All changes made in this file will be lost!

import sys
import os
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog

path = 'Mời bạn chọn video'
class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(673, 300)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(210, 0, 391, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.btnProcess = QtWidgets.QPushButton(self.centralwidget)
        self.btnProcess.setGeometry(QtCore.QRect(10, 110, 121, 41))
        self.btnProcess.setObjectName("btnProcess")
        self.hospi = QtWidgets.QTextEdit(self.centralwidget)
        self.hospi.setGeometry(QtCore.QRect(500, 80, 141, 31))
        self.hospi.setObjectName("hospi")
        self.label_8 = QtWidgets.QLabel(self.centralwidget)
        self.label_8.setGeometry(QtCore.QRect(180, 120, 31, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_8.setFont(font)
        self.label_8.setObjectName("label_8")
        self.line_3 = QtWidgets.QFrame(self.centralwidget)
        self.line_3.setGeometry(QtCore.QRect(460, 60, 181, 20))
        self.line_3.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_3.setObjectName("line_3")
        self.addr = QtWidgets.QTextEdit(self.centralwidget)
        self.addr.setEnabled(True)
        self.addr.setGeometry(QtCore.QRect(250, 160, 121, 31))
        self.addr.setObjectName("addr")
        self.age = QtWidgets.QTextEdit(self.centralwidget)
        self.age.setGeometry(QtCore.QRect(500, 120, 141, 31))
        self.age.setObjectName("age")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(150, 80, 81, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setGeometry(QtCore.QRect(440, 120, 31, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_7.setFont(font)
        self.label_7.setObjectName("label_7")
        self.ID = QtWidgets.QTextEdit(self.centralwidget)
        self.ID.setGeometry(QtCore.QRect(250, 80, 121, 31))
        self.ID.setObjectName("ID")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(430, 80, 61, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setEnabled(False)
        self.label.setGeometry(QtCore.QRect(360, 50, 111, 20))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label.setFont(font)
        self.label.setTextFormat(QtCore.Qt.RichText)
        self.label.setObjectName("label")
        self.label_10 = QtWidgets.QLabel(self.centralwidget)
        self.label_10.setGeometry(QtCore.QRect(170, 160, 41, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_10.setFont(font)
        self.label_10.setObjectName("label_10")
        self.btnSave = QtWidgets.QPushButton(self.centralwidget)
        self.btnSave.setGeometry(QtCore.QRect(470, 210, 75, 41))
        self.btnSave.setObjectName("btnSave")
        self.name = QtWidgets.QTextEdit(self.centralwidget)
        self.name.setGeometry(QtCore.QRect(250, 120, 121, 31))
        self.name.setObjectName("name")
        self.line_4 = QtWidgets.QFrame(self.centralwidget)
        self.line_4.setGeometry(QtCore.QRect(150, 60, 191, 20))
        self.line_4.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_4.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_4.setObjectName("line_4")
        self.btnChose = QtWidgets.QPushButton(self.centralwidget)
        self.btnChose.setGeometry(QtCore.QRect(10, 10, 121, 41))
        self.btnChose.setObjectName("btnChose")
        self.btnCancel = QtWidgets.QPushButton(self.centralwidget)
        self.btnCancel.setGeometry(QtCore.QRect(570, 210, 71, 41))
        self.btnCancel.setObjectName("btnCancel")
        self.btnVideo = QtWidgets.QPushButton(self.centralwidget)
        self.btnVideo.setGeometry(QtCore.QRect(10, 160, 121, 41))
        self.btnVideo.setObjectName("btnVideo")
        self.btnChart = QtWidgets.QPushButton(self.centralwidget)
        self.btnChart.setGeometry(QtCore.QRect(10, 210, 121, 41))
        self.btnChart.setObjectName("btnChart")
        self.btnCalib = QtWidgets.QPushButton(self.centralwidget)
        self.btnCalib.setGeometry(QtCore.QRect(10, 60, 121, 41))
        self.btnCalib.setObjectName("btnCalib")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 673, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_2.setText(_translate("MainWindow", "VIDEO VÀ ĐỒ THỊ THEO THĂNG BẰNG Ở NGƯỜI"))
        self.btnProcess.setText(_translate("MainWindow", "Process"))
        self.label_8.setText(_translate("MainWindow", "Tên"))
        self.label_6.setText(_translate("MainWindow", "Mã bệnh nhân"))
        self.label_7.setText(_translate("MainWindow", "Tuổi"))
        self.label_3.setText(_translate("MainWindow", "Bệnh viện"))
        self.label.setText(_translate("MainWindow", "Information"))
        self.label_10.setText(_translate("MainWindow", "Địa chỉ"))
        self.btnSave.setText(_translate("MainWindow", "Save"))
        self.btnChose.setText(_translate("MainWindow", "Chose video"))
        self.btnCancel.setText(_translate("MainWindow", "Cancel"))
        self.btnVideo.setText(_translate("MainWindow", "Show video detection"))
        self.btnChart.setText(_translate("MainWindow", "Show chart line"))
        self.btnCalib.setText(_translate("MainWindow", "Calib"))

        self.btnChose.clicked.connect(self.btnChose_handler)
        self.btnCalib.clicked.connect(self.btnCalib_handler)
        self.btnProcess.clicked.connect(self.btnProcess_handler)
        self.btnVideo.clicked.connect(self.btnVideo_handler)
        self.btnChart.clicked.connect(self.btnChart_handler)
        self.btnCancel.clicked.connect(self.btnCancel_handler)

        MainWindow.setWindowTitle(_translate("MainWindow", "Video và đồ thị theo dõi thăng bằng người"))

    def btnCancel_handler(self):
        print("Press cancel")
        sys.exit(app.exec_())

    def btnChose_handler(self):
        print("Press chose")
        self.open_dialog_box()
        print(path)

    def open_dialog_box(self):
        filename = QFileDialog.getOpenFileName()
        global path
        path = filename[0]

    def btnCalib_handler(self):
        print("Press calib")
        print("Press 'q' to quit")
        os.system('python Calib.py --video %s' % path)

    def btnChart_handler(self):
        print("Press Show chart line")
        print("Show line chart with base line is mean value")
        os.system('python CalculateFromExcelFile.py')

    def btnProcess_handler(self):
        print("Press process")
        print("Make image data, .exel for video, chart line")
        os.system('python PerformFromVideo.py')

    def btnVideo_handler(self):
        print("Press show video detection")
        print("Show video detect white paper")
        print("Please waitting video")
        os.system('python MakeVideo.py')




if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
