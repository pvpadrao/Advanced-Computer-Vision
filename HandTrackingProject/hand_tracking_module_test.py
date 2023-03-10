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
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 0), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)