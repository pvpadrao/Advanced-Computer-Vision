import cv2
import mediapipe as mp
import face_detection_module as FaceDetectionModule
import time

cap = cv2.VideoCapture(0)
detector = FaceDetectionModule.faceDetector()
previous_time = 0
while True:
    sucess, img = cap.read()
    img, bounding_boxes_list = detector.findFace(img)
    # calculating fps and output to window
    current_time = time.time()
    fps = 1 / (current_time - previous_time)
    previous_time = current_time
    cv2.putText(img, f'fps: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
    cv2.imshow("Image", img)
    cv2.waitKey(1)