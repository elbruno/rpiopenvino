# standard face detection sample with FPS in console

import face_recognition
import cv2
import time


video_capture = cv2.VideoCapture(0)

while True:
    start_time = time.time()
    ret, frame = video_capture.read()
    rgb_frame = frame[:, :, ::-1]

    face_locations = face_recognition.face_locations(rgb_frame)

    for top, right, bottom, left in face_locations:
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
    cv2.imshow('Video', frame)

    print("FPS: ", 1.0 / (time.time() - start_time)) # FPS = 1 / time to process loop

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()