# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '.\untitled.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1127, 796)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.select_folder = QtWidgets.QPushButton(self.centralwidget)
        self.select_folder.setGeometry(QtCore.QRect(40, 380, 301, 61))
        self.select_folder.setObjectName("select_folder")
        self.show_result = QtWidgets.QPushButton(self.centralwidget)
        self.show_result.setGeometry(QtCore.QRect(40, 460, 301, 61))
        self.show_result.setObjectName("show_result")
        self.ori_img = QtWidgets.QLabel(self.centralwidget)
        self.ori_img.setGeometry(QtCore.QRect(30, 50, 391, 311))
        self.ori_img.setText("")
        self.ori_img.setObjectName("ori_img")
        self.det_img = QtWidgets.QLabel(self.centralwidget)
        self.det_img.setGeometry(QtCore.QRect(440, 50, 391, 311))
        self.det_img.setText("")
        self.det_img.setObjectName("det_img")
        self.seg_img = QtWidgets.QLabel(self.centralwidget)
        self.seg_img.setGeometry(QtCore.QRect(440, 380, 391, 311))
        self.seg_img.setText("")
        self.seg_img.setObjectName("seg_img")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(200, 20, 171, 21))
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(620, 30, 171, 16))
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(600, 700, 161, 16))
        self.label_6.setObjectName("label_6")
        self.cur_img = QtWidgets.QLabel(self.centralwidget)
        self.cur_img.setGeometry(QtCore.QRect(890, 90, 191, 16))
        self.cur_img.setObjectName("cur_img")
        self.img_name = QtWidgets.QLabel(self.centralwidget)
        self.img_name.setGeometry(QtCore.QRect(890, 130, 211, 16))
        self.img_name.setObjectName("img_name")
        self.gt = QtWidgets.QLabel(self.centralwidget)
        self.gt.setGeometry(QtCore.QRect(890, 170, 201, 16))
        self.gt.setObjectName("gt")
        self.pred = QtWidgets.QLabel(self.centralwidget)
        self.pred.setGeometry(QtCore.QRect(890, 210, 211, 16))
        self.pred.setObjectName("pred")
        self.iou = QtWidgets.QLabel(self.centralwidget)
        self.iou.setGeometry(QtCore.QRect(890, 250, 211, 16))
        self.iou.setObjectName("iou")
        self.dc = QtWidgets.QLabel(self.centralwidget)
        self.dc.setGeometry(QtCore.QRect(890, 290, 221, 16))
        self.dc.setObjectName("dc")
        self.label_10 = QtWidgets.QLabel(self.centralwidget)
        self.label_10.setGeometry(QtCore.QRect(890, 420, 231, 20))
        self.label_10.setObjectName("label_10")
        self.ap_uc = QtWidgets.QLabel(self.centralwidget)
        self.ap_uc.setGeometry(QtCore.QRect(890, 460, 221, 16))
        self.ap_uc.setObjectName("ap_uc")
        self.ap_ue = QtWidgets.QLabel(self.centralwidget)
        self.ap_ue.setGeometry(QtCore.QRect(890, 500, 181, 16))
        self.ap_ue.setObjectName("ap_ue")
        self.ap_sc = QtWidgets.QLabel(self.centralwidget)
        self.ap_sc.setGeometry(QtCore.QRect(890, 540, 201, 16))
        self.ap_sc.setObjectName("ap_sc")
        self.previous = QtWidgets.QPushButton(self.centralwidget)
        self.previous.setGeometry(QtCore.QRect(40, 540, 91, 71))
        self.previous.setObjectName("previous")
        self.next = QtWidgets.QPushButton(self.centralwidget)
        self.next.setGeometry(QtCore.QRect(250, 540, 91, 71))
        self.next.setObjectName("next")
        self.fps = QtWidgets.QLabel(self.centralwidget)
        self.fps.setGeometry(QtCore.QRect(890, 330, 191, 16))
        self.fps.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(890, 580, 181, 16))
        self.label_2.setObjectName("label_2")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1127, 21))
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
        self.select_folder.setText(_translate("MainWindow", "Select Image Folder"))
        self.show_result.setText(_translate("MainWindow", "Detection and Segmentation"))
        self.label_4.setText(_translate("MainWindow", "Original Image"))
        self.label_5.setText(_translate("MainWindow", "Detection Result"))
        self.label_6.setText(_translate("MainWindow", "Segamentation Result"))
        self.cur_img.setText(_translate("MainWindow", "Current Image: "))
        self.img_name.setText(_translate("MainWindow", "Image Name: "))
        self.gt.setText(_translate("MainWindow", "GT: "))
        self.pred.setText(_translate("MainWindow", "Prediction: "))
        self.iou.setText(_translate("MainWindow", "Average IoU: "))
        self.dc.setText(_translate("MainWindow", "Dice Coefficient: "))
        self.label_10.setText(_translate("MainWindow", "Folder Mean Evaluation Metric"))
        self.ap_uc.setText(_translate("MainWindow", "AP50(uncover): "))
        self.ap_ue.setText(_translate("MainWindow", "AP50(uneven): "))
        self.ap_sc.setText(_translate("MainWindow", "AP50(scratch)"))
        self.previous.setText(_translate("MainWindow", "Previous"))
        self.next.setText(_translate("MainWindow", "Next"))
        self.fps.setText(_translate("MainWindow", "FPS: "))
        self.label_2.setText(_translate("MainWindow", "Average Dice Coefficient: "))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
