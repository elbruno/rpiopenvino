# This is the official sample demo file desribed in the installer documentation 
# Date: 2020 01 26 
# Install OpenVINO™ toolkit for Raspbian* OS 
# http://docs.openvinotoolkit.org/2019_R1/_docs_install_guides_installing_openvino_raspbian.html

import cv2 as cv
import sys, getopt

# Load the model.
net = cv.dnn.readNet('face-detection-adas-0001.xml',
                     'face-detection-adas-0001.bin')
# Specify target device.
net.setPreferableTarget(cv.dnn.DNN_TARGET_MYRIAD)
# Read an image.
imageToAnalyze = sys.argv[1]
print(f'image to analyze:{imageToAnalyze}')
frame = cv.imread(imageToAnalyze)
# Prepare input blob and perform an inference.
blob = cv.dnn.blobFromImage(frame, size=(672, 384), ddepth=cv.CV_8U)
net.setInput(blob)
out = net.forward()
# Draw detected faces on the frame.
for detection in out.reshape(-1, 7):
    confidence = float(detection[2])
    xmin = int(detection[3] * frame.shape[1])
    ymin = int(detection[4] * frame.shape[0])
    xmax = int(detection[5] * frame.shape[1])
    ymax = int(detection[6] * frame.shape[0])
    if confidence > 0.5:
        cv.rectangle(frame, (xmin, ymin), (xmax, ymax), color=(0, 255, 0))
# Save the frame to an image file.
cv.imwrite('out.png', frame)
