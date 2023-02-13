from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from ui import Ui_MainWindow
import sys
app = QtWidgets.QApplication(sys.argv)
MainWindow = QtWidgets.QMainWindow()

import cv2
import numpy as np
import mediapipe as mp
import pyrealsense2 as rs
from utils import cal_coordinate
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

from threading import Thread as th

class main(Ui_MainWindow):
    def __init__(self) -> None:
        super().setupUi(MainWindow)
        self.initialCameraRS()
        self.stopthread = False
        self.countvalue = 0
        self.stage = None
        self.signalSetup()

    def signalSetup(self):
        self.actionExit.triggered.connect(self.exitApp)

    def exitApp(self):
        self.stopthread = True
        self.pipeline.stop()
        sys.exit()

    def initialCameraRS(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        self.pipeline_profile = self.config.resolve(self.pipeline_wrapper)
        self.device_rs = self.pipeline_profile.get_device()
        self.device_product_line = str(self.device_rs.get_info(rs.camera_info.product_line))
        found_rgb = False
        for s in self.device_rs.sensors:
            if s.get_info(rs.camera_info.name) == 'RGB Camera':
                found_rgb = True
                break
        if not found_rgb:
            print("The demo requires Depth camera with Color sensor")
            exit(0)
        
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        if self.device_product_line == 'L500':
            self.config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
        else:
            self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        self.pipeline.start(self.config)
        print('Initial Realsense Success . . . !')

    def streaming(self):
        while True:
            if not self.stopthread:
                self.frames = self.pipeline.wait_for_frames()
                self.color_frame = self.frames.get_color_frame()
                if not self.color_frame:
                    continue
                self.image = np.asanyarray(self.color_frame.get_data())
                self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
                self.image = cv2.flip(self.image, 1)
                
                # Detection
                with mp_pose.Pose(
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5) as pose:
                    self.image.flags.writeable = False
                    # make detection
                    self.results = pose.process(self.image)
                    # Draw the pose annotation on the image.
                    self.image.flags.writeable = True
                    # Extract landmarks
                try:
                    landmarks = self.results.pose_landmarks.landmark
                    # Get coordinates
                    shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                    
                    # Calculate angle
                    angle = cal_coordinate(shoulder, elbow, wrist)

                    cv2.putText(self.image, str(angle), 
                                    tuple(np.multiply(elbow, [640, 480]).astype(int)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                            )
                        
                    # Curl counter logic
                    if angle > 160:
                        self.stage = "down"
                    if angle < 30 and self.stage == 'down':
                        self.stage = "up"
                        self.countvalue += 1
                except:
                    pass
                
                # Set Result image with pixmap
                try:
                    height, width, channel = self.image.shape
                    step = channel * width
                    qImg = QImage(self.image.data, width, height, step, QImage.Format_RGB888)
                    self.stream.setPixmap(QPixmap.fromImage(qImg))
                except:
                    self.stream.setText('Error for load streaming !')

                # Set Counter
                self.counter.display(self.countvalue)

            else:
                break

obj = main()

# Thread for detect video
class th1(th):
    def __init__(self):
        th.__init__(self)
    def run(self):
        if not obj.stopthread:
            obj.streaming()

th1th = th1()
th1th.start()

if __name__ == "__main__":
    MainWindow.show()
    ret = app.exec_()
    sys.exit(ret)