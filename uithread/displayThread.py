import os
import sys
import time

import cv2
import torch
import numpy as np

import pyrealsense2 as rs

import torch.nn.functional as nnF
from torchvision.transforms import functional as F

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QImage

from funcs.face_haar import haarFaceDetect
from funcs.fer_models import createFERmodel


class DisplayThread(QThread):
    updateFrameSignal = Signal(QImage, str)
    threadQMessageBoxSignal = Signal(str)  # emit info to main window QMessageBox
    threadStatusBarSignal = Signal(str)  # emit info to main window status bar
    logitsSignal = Signal(list)

    def __init__(self, face_detect=False, fer_detect=False, fer_modelname="vgg16", parent=None):
        QThread.__init__(self, parent)

        # mode
        self.use_rgb = True
        self.use_rgbd = False

        # camera setting
        self.cap = True
        self.capid = 0

        # face
        self.face_detect = face_detect
        self.fer_detect = face_detect and fer_detect
        self.fer_modelname = fer_modelname

        # 2d settings
        self.haar_file = None
        self.fer_2d_method = None

        # 3d settings
        self.enhance_method = None
        self.fer_3d_method = None

        # control
        self.status = True

        self.w = 640
        self.h = 480

        self.face_w = 224
        self.face_h = 224

        self.faceDetModel = None
        self.ferDetModel = None

        self.cuda_available = False
        self.check_cuda()

        self.class_label = ['NA', 'AN', 'DI', 'FE', 'HA', 'SA', 'SU']

        if self.face_detect:
            self.faceDetModel = haarFaceDetect()

        if self.fer_detect:
            # self.ferDetModel = createFERmodel(self.fer_modelname, weights_dir="./weights/ori_vgg16.pth", cuda=self.cuda_available)
            self.ferDetModel = createFERmodel(self.fer_modelname, weights_dir="./weights/vgg16_109lab.pth", cuda=self.cuda_available)

    def set_2d_methods(self, haar_filename, fer_2d_method):
        self.haar_file = os.path.join(cv2.data.haarcascades, haar_filename)  # haar file for face detection
        self.fer_2d_method = fer_2d_method

        if self.fer_2d_method == "None":
            self.fer_detect = False
        else:
            self.fer_detect = True

    def set_3d_methods(self, enhance_method, fer_3d_method):
        self.enhance_method = enhance_method
        self.fer_3d_method = fer_3d_method

        if self.fer_3d_method == "None":
            self.fer_detect = False
        else:
            self.fer_detect = True

    def set_camera(self, capid, reso):
        # set camera id
        self.capid = int(capid)

        # set resolution
        if reso == "240 x 320":
            self.w = 320
            self.h = 240
        elif reso == "480 x 640":
            self.w = 640
            self.h = 480
        elif reso == "480 x 720 (DV)":
            self.w = 720
            self.h = 480
        elif reso == "720 x 1280 (720p)":
            self.w = int(1280/2)
            self.h = int(720/2)
        elif reso == "1080 x 1920 (1080p)":
            self.w = int(1920 / 4)
            self.h = int(1080 / 4)
        else:
            raise Exception("Resolution does not match.")

        return

    def set_mode(self, use_rgb, use_rgbd):
        self.use_rgb = use_rgb
        self.use_rgbd = use_rgbd
        return

    def check_cuda(self):
        if torch.cuda.is_available():
            self.cuda_available = True
        else:
            self.cuda_available = False

        return

    def check_rgbd(self):
        """ Check RGB-D camera.  (device: Intel RealSense)
        """

        # Configure depth and color streams
        pipeline = rs.pipeline()
        config = rs.config()

        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(pipeline)

        try:
            pipeline_profile = config.resolve(pipeline_wrapper)
        except:
            realsense_available = False
            return realsense_available, "No RealSense device."
        else:
            # double check the RealSense RGB camera
            device = pipeline_profile.get_device()
            found_rgb = False
            for s in device.sensors:
                if s.get_info(rs.camera_info.name) == 'RGB Camera':
                    found_rgb = True

            if found_rgb:
                realsense_available = True
                device = pipeline_profile.get_device()
                device_product_line = str(device.get_info(rs.camera_info.product_line))

                return realsense_available, device_product_line
            else:
                realsense_available = False
                return realsense_available, "RealSense device checked.\nBUT, abnormal RGB camera."

    def run(self):

        # RGB mode
        if self.use_rgb and not self.use_rgbd:
            self.run_rgb()

        # RGB-D mode
        if self.use_rgbd:
            self.run_rgbd()

        # sys.exit(-1)
        return

    def run_rgb(self):
        """ using RGB camera device. """
        self.status = True
        self.cap = cv2.VideoCapture(self.capid, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.h)

        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        try:
            while self.status:

                ret, frame = self.cap.read()
                if not ret:
                    self.threadQMessageBoxSignal.emit("No 2D video signal. Please change the Camera ID, or check the webcam.")
                    break

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # 2D face detection (you can use any detections here)
                face_exist, frame, face_frame, face_bbox = self.face_detection(frame)

                # Facial expression recognition
                if self.fer_detect and face_exist:
                    fer_logits = self.fer_detection(face_frame)

                    logits_na = fer_logits[0]
                    logits_an = fer_logits[1]
                    logits_di = fer_logits[2]
                    logits_fe = fer_logits[3]
                    logits_ha = fer_logits[4]
                    logits_sa = fer_logits[5]
                    logits_su = fer_logits[6]

                    self.logitsSignal.emit([logits_na, logits_an, logits_di, logits_fe, logits_ha, logits_sa, logits_su])
        finally:
            face_img = QImage("./resource/camera2d.png").scaled(int(1920 / 4), int(1080 / 4), mode=Qt.SmoothTransformation)
            self.updateFrameSignal.emit(face_img, "2d")
            self.logitsSignal.emit([0, 0, 0, 0, 0, 0, 0])

            self.cap.release()
            cv2.destroyAllWindows()

            time.sleep(1)  # waiting for emitting all signals

        return

    def run_rgbd(self):
        """ using RGB-D Intel RealSense device. """

        # Configure depth and color streams
        pipeline = rs.pipeline()
        config = rs.config()

        # Set fixed resolution: 640 x 480, for both RGB and D
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # Start streaming
        pipeline.start(config)

        try:
            while self.status:
                # Wait for a coherent pair of frames: depth and color
                frames = pipeline.wait_for_frames()
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                if not depth_frame or not color_frame:
                    continue

                # Convert images to numpy arrays
                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())

                depth_image = cv2.flip(depth_image, 1)

                color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

                # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
                valid_meter = 1.5
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=255 / (valid_meter * 1000)),
                                                   cv2.COLORMAP_JET)

                # convert depth map to disparity map
                disp_image = depth_image
                disp_image[disp_image > 1000] = 1000

                disp_image = 255 - disp_image / disp_image.max() * 255
                disp_image = disp_image.astype("uint8")
                disp_image = np.expand_dims(disp_image, axis=2).repeat(repeats=3, axis=2)

                # 2D face detection (you can use any detections here)
                face_exist, frame, face_frame, face_bbox = self.face_detection(color_image)

                if face_exist:
                    self.face_3d_draw(depth_colormap, face_bbox)  # show depth map, or use 'disp_image' for disparity map

                    # face_frame_3d = self.face_3d_crop(depth_image)
                else:
                    self.face_3d_draw(depth_colormap)

                # Facial expression recognition
                if self.fer_detect and face_exist:
                    fer_logits = self.fer_detection(face_frame)

                    logits_na = fer_logits[0]
                    logits_an = fer_logits[1]
                    logits_di = fer_logits[2]
                    logits_fe = fer_logits[3]
                    logits_ha = fer_logits[4]
                    logits_sa = fer_logits[5]
                    logits_su = fer_logits[6]

                    self.logitsSignal.emit(
                        [logits_na, logits_an, logits_di, logits_fe, logits_ha, logits_sa, logits_su])

        finally:
            # Stop streaming
            pipeline.stop()

            # reset QMainWindow
            face_img = QImage("./resource/camera2d.png").scaled(int(1920 / 4), int(1080 / 4), mode=Qt.SmoothTransformation)
            self.updateFrameSignal.emit(face_img, "2d")

            face_img = QImage("./resource/camera3d.png").scaled(int(1920 / 4), int(1080 / 4), mode=Qt.SmoothTransformation)
            self.updateFrameSignal.emit(face_img, "3d")

            self.logitsSignal.emit([0, 0, 0, 0, 0, 0, 0])

            cv2.destroyAllWindows()

        return

    def face_detection(self, frame):
        """ Running 2D face detection.

        :param
            frame: (np.ndarray) raw input frame.

        :return:
            face_exist: (bool) True: face is detected.
            frame: (np.ndarray) raw input frame with bounding box.
            face_frame: (np.ndarray or NoneType) cropped frame only for face.
        """

        face_exist = False
        face_frame = None

        # Horizontal flip & Scaling
        frame = cv2.flip(frame, 1)

        if self.face_detect:
            face_exist, frame, face_frame, face_bbox = self.faceDetModel.run(frame, self.haar_file)

        h, w, ch = frame.shape
        img = QImage(frame, w, h, ch * w, QImage.Format_RGB888)
        scaled_img = img.scaled(self.w, self.h, Qt.KeepAspectRatio)

        # Emit singal for updating mainWindow
        self.updateFrameSignal.emit(scaled_img, "2d")

        return face_exist, frame, face_frame, face_bbox

    def fer_detection(self, face_frame):

        # pre-processing (ps: normalization should be in CNN.forward(), due to different mean/std)
        face_frame = cv2.resize(face_frame, (self.face_w, self.face_h), interpolation=cv2.INTER_CUBIC)
        face_frame = F.to_tensor(face_frame)
        face_frame = face_frame.unsqueeze(0)

        logits = None
        with torch.no_grad():
            if self.cuda_available:
                self.threadStatusBarSignal.emit("FER detecting (CUDA) ...")
                self.ferDetModel = self.ferDetModel.cuda()
                logits = self.ferDetModel(face_frame.cuda())

            if not self.cuda_available:
                self.threadStatusBarSignal.emit("FER detecting (CPU) ...")
                logits = self.ferDetModel(face_frame)

        # Logits to probabilities
        probs = nnF.softmax(logits, dim=1)
        probs = probs[0].detach().cpu().numpy()

        return probs

    def face_3d_draw(self, faceimg, bbox=None):
        """ Draw 3D face bounding box by face bbox."""

        if bbox is not None:
            # Draw bbox
            # cv2.rectangle(faceimg, bbox[0], bbox[1], (0, 255, 0), 2)
            pass

        # Emit 3D image
        h, w, ch = faceimg.shape
        img = QImage(faceimg, w, h, ch * w, QImage.Format_RGB888)
        scaled_img = img.scaled(self.w, self.h, Qt.KeepAspectRatio)

        # Emit singal for updating mainWindow
        self.updateFrameSignal.emit(scaled_img, "3d")
        return

    def face_3d_crop(self):
        return