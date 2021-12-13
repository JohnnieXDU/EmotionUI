import os
import sys
import time

import cv2
import numpy as np

from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtGui import QAction, QImage, QKeySequence, QPixmap, QFont
from PySide6.QtWidgets import (QApplication, QComboBox, QGroupBox,
                               QHBoxLayout, QLabel, QMainWindow, QPushButton,
                               QSizePolicy, QVBoxLayout, QWidget, QMessageBox, QProgressBar,
                               QGridLayout, QStatusBar)

from uithread.displayThread import DisplayThread


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Expression flag
        # 0-6 indicates:0)Natural 1)Angry 2)Disgust 3)Fear 4)Happy 5)Sad 6)Surprise
        self.emoFlag = 0

        # Title and dimensions
        self.setWindowTitle("FER System")
        self.setGeometry(300, 300, 1200, 550)

        self.dis_w = int(1920 / 4)
        self.dis_h = int(1080 / 4)

        # Menu widgets setup
        self.menu = None
        self.menu_file = None
        self.act_exit = None
        self.menu_edit = None
        self.act_dummy = None
        self.menu_help = None
        self.act_require = None
        self.act_about = None

        # mode flag
        self.use_rgb = False
        self.use_rgbd = False

        # Progress bar
        self.progress_na = QProgressBar()
        self.progress_an = QProgressBar()
        self.progress_di = QProgressBar()
        self.progress_fe = QProgressBar()
        self.progress_ha = QProgressBar()
        self.progress_sa = QProgressBar()
        self.progress_su = QProgressBar()

        self.setupMenuBar()

        # Central widgets group

        # FER results area
        self.icon_default = QImage("./resource/icon_default.png").scaled(256 / 2, 256 / 2, mode=Qt.SmoothTransformation)
        self.icon_natural = QImage("./resource/icon_natural.png").scaled(256 / 2, 256 / 2, mode=Qt.SmoothTransformation)
        self.icon_angry = QImage("./resource/icon_angry.png").scaled(256 / 2, 256 / 2, mode=Qt.SmoothTransformation)
        self.icon_disgust = QImage("./resource/icon_disgust.png").scaled(256 / 2, 256 / 2, mode=Qt.SmoothTransformation)
        self.icon_fear = QImage("./resource/icon_fear.png").scaled(256 / 2, 256 / 2, mode=Qt.SmoothTransformation)
        self.icon_happy = QImage("./resource/icon_happy.png").scaled(256 / 2, 256 / 2, mode=Qt.SmoothTransformation)
        self.icon_sad = QImage("./resource/icon_sad.png").scaled(256 / 2, 256 / 2, mode=Qt.SmoothTransformation)
        self.icon_surprise = QImage("./resource/icon_surprise.png").scaled(256 / 2, 256 / 2, mode=Qt.SmoothTransformation)

        self.setupCentralWidgets()

        # Main status bar
        self.setupStatusBar()

        # 2D 3D video displaying thread
        self.dis_thread = DisplayThread(face_detect=True, fer_detect=True)
        self.dis_thread.exit(self.close())
        self.dis_thread.updateFrameSignal.connect(self.updateFrame)
        self.dis_thread.threadQMessageBoxSignal.connect(self.setQMessageBoxSlot)
        self.dis_thread.threadStatusBarSignal.connect(self.setStatusBarSlot)
        self.dis_thread.logitsSignal.connect(self.setProgressBar)
        self.dis_thread.logitsSignal.connect(self.setFERIcon)

        # Connections
        self.act_dummy.triggered.connect(self.dummySlot)
        self.act_require.triggered.connect(self.showRequireSlot)
        self.act_about.triggered.connect(self.showAboutSlot)

        self.apply_2d_set_button.clicked.connect(self.applyTexDisSet)
        self.combo_2d_face_methods.currentTextChanged.connect(self.setFaceDetectModel)

        self.RGBD_button.clicked.connect(self.applyRGBDSet)

        self.start_button.clicked.connect(self.start)
        self.stop_button.clicked.connect(self.kill_thread)

        self.stop_button.setEnabled(False)


    def setupMenuBar(self):
        """ Setup the display layout for menu bar.
        """

        self.menu = self.menuBar()

        # File menu
        self.menu_file = self.menu.addMenu("File")
        self.act_exit = QAction("Exit", self, triggered=qApp.quit, shortcut=QKeySequence("Q"))
        self.menu_file.addAction(self.act_exit)

        # Edit menu
        self.menu_edit = self.menu.addMenu("Edit")

        self.act_dummy = QAction("EditDummy", self)
        self.menu_edit.addAction(self.act_dummy)

        # Help menu
        self.menu_help = self.menu.addMenu("Help")

        self.act_require = QAction("Requirements", self)
        self.menu_help.addAction(self.act_require)

        self.act_about = QAction("About", self)
        self.menu_help.addAction(self.act_about)

    def setupCentralWidgets(self):
        """ Setup the display layout for central widgets.
        """

        # ==> Step 1. Create local widgets

        # 1)Software main title widgets
        self.title = QLabel()
        self.title.setText("<font size=20><b>Multi-Modality Facial Expression Recognition System</b></font>")
        self.title.setAlignment(Qt.AlignCenter)

        # 2) widgets for 2D texture stream
        self.title_2d = QLabel()  # 2D title
        self.title_2d.setText("<font size=5>Texture Stream</font>")
        self.title_2d.setAlignment(Qt.AlignCenter)

        self.display_2d = QLabel()  # 2D display
        self.display_2d.setFixedSize(self.dis_w, self.dis_h)

        face_img = QImage("./resource/camera2d.png").scaled(self.dis_w, self.dis_h, mode=Qt.SmoothTransformation)
        self.display_2d.setPixmap(QPixmap.fromImage(face_img))

        # 2D Display settings
        layout_2d_display_settings = QHBoxLayout()

        # Add 2D camera setting
        layout_2d_display_settings.addSpacing(10)
        layout_2d_display_settings.addWidget(QLabel("Camera Id."))
        self.combo_2d_cam_id = QComboBox()
        camids = ["0", "1", "2", "3", "4", "5"]
        for camid in camids:
            self.combo_2d_cam_id.addItem(camid)

        self.combo_2d_cam_id.setCurrentIndex(2)
        layout_2d_display_settings.addWidget(self.combo_2d_cam_id)
        layout_2d_display_settings.addSpacing(30)

        # Add resolution-selection
        layout_2d_display_settings.addWidget(QLabel("Resolution"))
        self.combo_2d_reso = QComboBox()
        resos = ["240 x 320", "480 x 640", "480 x 720 (DV)", "720 x 1280 (720p)", "1080 x 1920 (1080p)"]
        for reso in resos:
            self.combo_2d_reso.addItem(reso)

        self.combo_2d_reso.setCurrentIndex(1)  # 480 x 640 as default

        layout_2d_display_settings.addWidget(self.combo_2d_reso)

        self.apply_2d_set_button = QPushButton("Apply RGB")

        layout_2d_display_settings.addSpacing(26)
        layout_2d_display_settings.addWidget(self.apply_2d_set_button)

        layout_2d_display_settings.addStretch()
        # layout_2d_display_settings.addSpacing(100)

        # 2D methods box
        self.methods_2d_widget = QGroupBox("Choose 2D Methods")  # 2D methods groupBox
        self.methods_2d_widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

        layout_2d_methods = QVBoxLayout()

        # Face detection
        layout_2d_face_methods = QHBoxLayout()
        self.combo_2d_face_methods = QComboBox()
        methods_2d_face = os.listdir(cv2.data.haarcascades)
        for method in methods_2d_face:
            if method.endswith(".xml"):
                self.combo_2d_face_methods.addItem(method)

        self.combo_2d_face_methods.setCurrentText("haarcascade_frontalface_alt2.xml")
        layout_2d_face_methods.addWidget(QLabel("Face Detection"), 10)
        layout_2d_face_methods.addWidget(self.combo_2d_face_methods, 90)

        # 2D FER Nets
        layout_2d_fer_methods = QHBoxLayout()
        self.combo_2d_fer_methods = QComboBox()
        methods_2d_fer = ['None', 'LeNet', 'VGG', 'ResNet', 'ViT', 'FA-CNN']
        for method in methods_2d_fer:
            self.combo_2d_fer_methods.addItem(method)

        self.combo_2d_fer_methods.setCurrentIndex(2)  # default VGG

        layout_2d_fer_methods.addWidget(QLabel("2D FER Nets "), 10)
        layout_2d_fer_methods.addWidget(self.combo_2d_fer_methods, 90)

        layout_2d_methods.addLayout(layout_2d_face_methods)
        layout_2d_methods.addLayout(layout_2d_fer_methods)

        self.methods_2d_widget.setLayout(layout_2d_methods)

        self.texture_module_layout = QGridLayout()
        self.texture_module_layout.addWidget(self.title_2d, 0, 0, 2, 10)
        self.texture_module_layout.addWidget(self.display_2d, 2, 0, 3, 10)
        self.texture_module_layout.addLayout(layout_2d_display_settings, 5, 0, 1, 10)
        self.texture_module_layout.addWidget(self.methods_2d_widget, 6, 0, 1, 10)

        # 3) widgets for 3D depth stream
        self.title_3d = QLabel()  # 3D title
        self.title_3d.setText("<font size=5>Depth Stream</font>")
        self.title_3d.setAlignment(Qt.AlignCenter)

        self.display_3d = QLabel()  # 3D display
        self.display_3d.setFixedSize(self.dis_w, self.dis_h)

        face_img = QImage("./resource/camera3d.png").scaled(self.dis_w, self.dis_h, mode=Qt.SmoothTransformation)
        self.display_3d.setPixmap(QPixmap.fromImage(face_img))

        # 3D settings
        layout_3d_display_settings = QHBoxLayout()

        # RGB-D camera
        self.RGBD_button = QPushButton("Apply RGB-D")
        layout_3d_display_settings.addWidget(self.RGBD_button)

        # 3D methods box
        self.methods_3d_widget = QGroupBox("Choose 3D Methods")  # 2D methods groupBox
        self.methods_3d_widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

        layout_3d_methods = QVBoxLayout()

        # 3D enhancement methods
        layout_3d_enh_methods = QHBoxLayout()
        self.combo_3d_enh_methods = QComboBox()
        methods_3d_enh = ['Log', 'HE', 'GEMax']
        for method in methods_3d_enh:
            self.combo_3d_enh_methods.addItem(method)

        layout_3d_enh_methods.addWidget(QLabel("Enhancement"), 10)
        layout_3d_enh_methods.addWidget(self.combo_3d_enh_methods, 90)

        # 3D FER Nets
        layout_3d_fer_methods = QHBoxLayout()
        self.combo_3d_fer_methods = QComboBox()
        methods_3d_fer = ['None', 'LeNet', 'VGG', 'ResNet', 'ViT', 'FA-CNN3D']
        for method in methods_3d_fer:
            self.combo_3d_fer_methods.addItem(method)

        self.combo_3d_fer_methods.setCurrentIndex(2)  # default VGG

        layout_3d_fer_methods.addWidget(QLabel("3D FER Nets "), 10)
        layout_3d_fer_methods.addWidget(self.combo_3d_fer_methods, 90)

        layout_3d_methods.addLayout(layout_3d_enh_methods)
        layout_3d_methods.addLayout(layout_3d_fer_methods)

        self.methods_3d_widget.setLayout(layout_3d_methods)

        depth_module_layout = QGridLayout()
        depth_module_layout.addWidget(self.title_3d, 0, 0, 2, 10)
        depth_module_layout.addWidget(self.display_3d, 2, 0, 3, 10)
        depth_module_layout.addLayout(layout_3d_display_settings, 5, 0, 1, 10)
        depth_module_layout.addWidget(self.methods_3d_widget, 6, 0, 1, 10)

        # 4) Result widgets
        # - i) result title
        self.title_res = QLabel()
        self.title_res.setText("<font size=5>Predicted Expression</font>")
        self.title_res.setAlignment(Qt.AlignCenter)

        # - ii) display result image
        self.display_res = QLabel()
        self.display_res.setFixedSize(256/2, 256/2)

        self.display_res.setPixmap(QPixmap.fromImage(self.icon_default))
        self.display_res.setAlignment(Qt.AlignCenter)

        # - iii) class name and probability bar
        name_na = QLabel("Nature")
        name_an = QLabel("Angry")
        name_di = QLabel("Disgust")
        name_fe = QLabel("Fear")
        name_ha = QLabel("Happy")
        name_sa = QLabel("Sad")
        name_su = QLabel("Surprise")

        self.progress_na.setFixedWidth(196)
        self.progress_an.setFixedWidth(196)
        self.progress_di.setFixedWidth(196)
        self.progress_fe.setFixedWidth(196)
        self.progress_ha.setFixedWidth(196)
        self.progress_sa.setFixedWidth(196)
        self.progress_su.setFixedWidth(196)

        # - iv) predicted results layout
        self.res_group = QGroupBox("FER Probability Bar")
        res_layout = QGridLayout()

        res_layout.addWidget(name_na, 0, 0, 1, 1)
        res_layout.addWidget(name_an, 1, 0, 1, 1)
        res_layout.addWidget(name_di, 2, 0, 1, 1)
        res_layout.addWidget(name_fe, 3, 0, 1, 1)
        res_layout.addWidget(name_ha, 4, 0, 1, 1)
        res_layout.addWidget(name_sa, 5, 0, 1, 1)
        res_layout.addWidget(name_su, 6, 0, 1, 1)

        res_layout.addWidget(self.progress_na, 0, 2, 1, 9)
        res_layout.addWidget(self.progress_an, 1, 2, 1, 9)
        res_layout.addWidget(self.progress_di, 2, 2, 1, 9)
        res_layout.addWidget(self.progress_fe, 3, 2, 1, 9)
        res_layout.addWidget(self.progress_ha, 4, 2, 1, 9)
        res_layout.addWidget(self.progress_sa, 5, 2, 1, 9)
        res_layout.addWidget(self.progress_su, 6, 2, 1, 9)

        self.res_group.setLayout(res_layout)

        # 7) Start/Stop button
        self.group_buttons = QWidget()

        button_layout = QHBoxLayout()
        self.start_button = QPushButton("Start")
        self.stop_button = QPushButton("Stop")

        self.start_button.setFixedWidth(150)

        self.start_button.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.stop_button.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)

        self.group_buttons.setLayout(button_layout)

        # ==> Step 2. Setup main layout and central widget
        res_control_layout = QGridLayout()
        res_control_layout.addWidget(self.title_res, 0, 0, 1, 10)
        res_control_layout.addWidget(self.display_res, 1, 2, 1, 10)
        res_control_layout.addWidget(self.res_group, 2, 0, 1, 10)
        res_control_layout.addWidget(self.group_buttons, 3, 0, 1, 10)

        main_layout = QHBoxLayout()
        main_layout.addLayout(self.texture_module_layout)
        main_layout.addLayout(depth_module_layout)
        main_layout.addLayout(res_control_layout)

        central_layout = QVBoxLayout()
        central_layout.addWidget(self.title)
        central_layout.addLayout(main_layout)

        central_widget = QWidget(self)
        central_widget.setLayout(central_layout)
        self.setCentralWidget(central_widget)

        # ==> Step 3. Apply functions
        # Thread in charge of updating the 2D image
        # self.thread2D = Thread(self)  # need to implement
        # Thread in charge of updating the 3D image
        # self.thread3D = Thread(self)  # need to implement

        pass

    def setupStatusBar(self):
        """ Setup the display layout for status bar.
        """
        status_bar = QStatusBar(self)
        self.setStatusBar(status_bar)

        self.statusBar().showMessage("Read to go ...")

        return

    @Slot()
    def setFERIcon(self, logits):
        """ Set FER result icon.
        """
        expression = np.argmax(logits)
        expression_image = None
        if expression == 0:
            expression_image = QPixmap.fromImage(self.icon_natural)
        elif expression == 1:
            expression_image = QPixmap.fromImage(self.icon_angry)
        elif expression == 2:
            expression_image = QPixmap.fromImage(self.icon_disgust)
        elif expression == 3:
            expression_image = QPixmap.fromImage(self.icon_fear)
        elif expression == 4:
            expression_image = QPixmap.fromImage(self.icon_happy)
        elif expression == 5:
            expression_image = QPixmap.fromImage(self.icon_sad)
        elif expression == 6:
            expression_image = QPixmap.fromImage(self.icon_surprise)
        else:
            raise Exception("Error max logits.")

        self.display_res.setPixmap(expression_image)
        return


    @Slot()
    def setProgressBar(self, logits):
        """ Set progress bar, set the FER logits value.
        """

        self.progress_na.setValue(logits[0]*100)
        self.progress_an.setValue(logits[1]*100)
        self.progress_di.setValue(logits[2]*100)
        self.progress_fe.setValue(logits[3]*100)
        self.progress_ha.setValue(logits[4]*100)
        self.progress_sa.setValue(logits[5]*100)
        self.progress_su.setValue(logits[6]*100)
        return

    def resetProgressBar(self):
        """ Reset progress bar, clear the FER logits value.
        """

        self.progress_na.reset()
        self.progress_an.reset()
        self.progress_di.reset()
        self.progress_fe.reset()
        self.progress_ha.reset()
        self.progress_sa.reset()
        self.progress_su.reset()

        expression_image = QPixmap.fromImage(self.icon_default)
        self.display_res.setPixmap(expression_image)
        return

    @Slot()
    def dummySlot(self):
        QMessageBox.about(self,
                          "Dummy Slot",
                          "This is a dummy Slot function. Implement it in the future.")

    @Slot()
    def showRequireSlot(self):
        QMessageBox.about(self,
                          "Developing Env",
                          "The software is developed based on Python.\n\nRequirements:\n\n- Python 3.6 \n- Opencv-python 4.5 \n- PySide6 6.2.1 \n- PyTorch 0.3.1 cuda80\n- TorchVision 0.1.9\n\n"
                          )

    @Slot()
    def showAboutSlot(self):
        dev_info = "This software is developed as a byproduct of Yang's PhD thesis (2021). \n\nDeveloper: Yang Jiao \nEmail: yjiao.xdu@gmail.com \nXidian University \nJohns Hopkins University"
        ack_info = "Free to anyone. No commercial purpose."
        license_info = "GPL License."
        QMessageBox.about(self,
                          "Acknowledgement",
                          dev_info+"\n\n"+ack_info+"\n\n"+license_info
                          )

    @Slot()
    def setFaceDetectModel(self, method):
        self.dis_thread.set_file(method)
        return

    @Slot()
    def start(self):
        """ Start a uithread for displaying camera video.
        """

        self.statusBar().showMessage("Starting FER ...")
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

        if not self.use_rgb and not self.use_rgbd:
            self.applyTexDisSet()
            self.dis_thread.set_2d_methods(self.combo_2d_face_methods.currentText(),
                                           self.combo_2d_fer_methods.currentText())

        self.dis_thread.set_mode(self.use_rgb, self.use_rgbd)
        self.dis_thread.start()

        return

    @Slot()
    def kill_thread(self):
        """ Kill the uithread which is displaying the camera video.
        """

        self.statusBar().showMessage("Finishing FER ...")
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

        self.dis_thread.status = False
        time.sleep(2)  # waiting for emitting all signals

        self.dis_thread.terminate()
        time.sleep(1)

        self.statusBar().showMessage("Read to go ...")
        self.dis_thread.status = True

        return

    @Slot()
    def updateFrame(self, image, window="2d"):
        if window == "2d":
            self.display_2d.setPixmap(QPixmap.fromImage(image))

        if window == "3d":
            self.display_3d.setPixmap(QPixmap.fromImage(image))

        return

    @Slot()
    def setQMessageBoxSlot(self, something):

        self.statusBar().showMessage(something)
        QMessageBox.warning(self,
                          "RGB Signal",
                          something)

        return


    @Slot()
    def setStatusBarSlot(self, something):

        self.statusBar().showMessage(something)

        return

    @Slot()
    def applyTexDisSet(self):
        self.statusBar().showMessage("Applying RGB setting ...")
        camid = self.combo_2d_cam_id.currentText()
        reso = self.combo_2d_reso.currentText()

        self.dis_thread.set_camera(camid, reso)

        QMessageBox.about(self,
                          "RGB Apply",
                          "Camera Id = {}\nResolution = {}".format(camid, reso))
        self.statusBar().showMessage("CameraId = {}   Resolution = {}".format(camid, reso))

        self.use_rgb = True
        self.use_rgbd = False

        self.dis_thread.status = True
        self.dis_thread.set_2d_methods(self.combo_2d_face_methods.currentText(),
                                       self.combo_2d_fer_methods.currentText())

        return

    @Slot()
    def applyRGBDSet(self):
        self.statusBar().showMessage("Applying RGB-D setting ...")

        realsense_available, deviceid = self.dis_thread.check_rgbd()

        if realsense_available:
            self.statusBar().showMessage("Intel RealSense {} detected.".format(deviceid))
            QMessageBox.about(self,
                              "RGB-D Apply",
                              "Intel RealSense {} detected.\nIntel RealSense {} applied.".format(deviceid, deviceid))

        else:
            self.statusBar().showMessage("No RGB-D device detected.")
            QMessageBox.warning(self,
                              "RGB-D Apply",
                              "No RGB-D device detected.\nPlease connect Intel RealSense device.")

        self.use_rgb = False
        self.use_rgbd = True

        self.dis_thread.status = True
        self.dis_thread.set_2d_methods(self.combo_2d_face_methods.currentText(),
                                       self.combo_2d_fer_methods.currentText())
        self.dis_thread.set_3d_methods(self.combo_3d_enh_methods.currentText(),
                                       self.combo_3d_fer_methods.currentText())

        return

