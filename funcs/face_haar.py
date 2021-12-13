"""
This file contains Viola-Jones face detection method.
"""

import cv2
import copy


class haarFaceDetect():
    def __init__(self):
        self.scaleFactor = 1.1
        self.minNeighbors = 5
        self.minSize = (30, 30)
        self.face_exist = False

    def run(self, frame, haar_file):
        cascade = cv2.CascadeClassifier(haar_file)

        # Reading frame in gray scale to process the pattern
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        detections = cascade.detectMultiScale(gray_frame, scaleFactor=self.scaleFactor,
                                              minNeighbors=self.minNeighbors, minSize=self.minSize)

        # Drawing green rectangle around the pattern
        if len(detections) > 0:
            self.face_exist = True
            (x, y, w, h) = detections[0]  # only take the first major face
            pos_ori = (x, y)
            pos_end = (x + w, y + h)
            color = (0, 255, 0)

            # Crop face from detection
            face_frame = copy.deepcopy(frame[pos_ori[1]:pos_end[1], pos_ori[0]:pos_end[0], :])
            face_bbox = [pos_ori, pos_end]
            cv2.rectangle(frame, pos_ori, pos_end, color, 2)
        else:
            self.face_exist = False
            face_frame = None
            face_bbox = None

        return self.face_exist, frame, face_frame, face_bbox