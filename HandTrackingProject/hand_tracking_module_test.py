import cv2
import mediapipe as mp
import time
import hand_tracking_module as handTrackingModule

cap = cv2.VideoCapture(0)
# setting time
previous_time = 0
current_time = 0
detector = handTrackingModule.handDetector()
while True:
    sucess, img = cap.read()
    img = detector.findHands(img)
    landmark_list = detector.findHandLandmarkPosition(img)
    # checking if hand is detected
    if len(landmark_list) !=0:
        # tracking the thumb
        print(landmark_list[4])
    #calculting fps and printing to window
    current_time = time.time()
    fps = 1 / (current_time - previous_time)
    previous_time = current_time
    img = cv2.flip(img,1)
    cv2.putText(img, ("FPS: " + str(int(fps))), (10, 430), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
    cv2.imshow("Image", img)
    cv2.waitKey(1)