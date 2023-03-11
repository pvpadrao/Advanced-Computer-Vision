import cv2
import mediapipe as mp
import pose_estimation_module as poseEstimationModule


cap = cv2.VideoCapture("PoseEstimationProject/videos/defante.mp4")
detector = poseEstimationModule.poseDetector()
while True:
    sucess, img = cap.read()
    detector.findPose(img)
    #landmark_list = detector.findPosition(img)
    #print(landmark_list[14])
    img = cv2.resize(img, (300, 620)) 
    cv2.imshow("Image", img)
    cv2.waitKey(1)