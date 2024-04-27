import cv2
import os
import numpy as np

cap = cv2.VideoCapture("/mnt/extended/AIC24/test/scene_088/camera_0907/video.mp4")
ret, frame = cap.read()
cv2.imwrite("/mnt/extended/xzy/mmpose/examples/88.jpg",frame)

cap.release()