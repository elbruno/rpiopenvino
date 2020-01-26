# perform face detection
# display detected face frame
# display FPS info in webcam video feed

# This is the official sample demo file desribed in the installer documentation 
# Date: 2020 01 26 
# Install OpenVINO™ toolkit for Raspbian* OS 
# http://docs.openvinotoolkit.org/2019_R1/_docs_install_guides_installing_openvino_raspbian.html

import cv2
import time

# Load the model.
net = cv2.dnn.readNet('face-detection-adas-0001.xml',
                     'face-detection-adas-0001.bin')
# Specify target device.
net.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)

# open video frame
video_capture = cv2.VideoCapture(0)

while True:
    start_time = time.time()
    ret, frame = video_capture.read()

    # frame resize to improve performance
    frame = imutils.resize(frame, width=648, height=480)
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Prepare input blob and perform an inference.
    blob = cv2.dnn.blobFromImage(reg_frame, size=(640, 480), ddepth=cv2.CV_8U)
    net.setInput(blob)
    out = net.forward()

    # Draw detected faces on the frame.
    for detection in out.reshape(-1, 7):
        confidence = float(detection[2])
        xmin = int(detection[3] * frame.shape[1])
        ymin = int(detection[4] * frame.shape[0])
        xmax = int(detection[5] * frame.shape[1])
        ymax = int(detection[6] * frame.shape[0])

        print(f'xmin: {xmin} - ymin: {ymin} - xmax: {xmax} - ymax: {ymax}')

        if confidence > 0.5:
            cv.rectangle(frame, (xmin, ymin), (xmax, ymax), color=(0, 255, 0))           
    
    #display FPS
    fpsInfo = "FPS: " + str(1.0 / (time.time() - start_time)) # FPS = 1 / time to process loop
    print(fpsInfo)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(frame, fpsInfo, (10, 20), font, 0.4, (255, 255, 255), 1)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()