import cv2
import mediapipe as mp
import numpy as np


class poseDetector():
    def __init__(self, mode=False, model_complexity = 1, smooth= True, upperBody = False, detection_confidence = 0.5, 
                 track_confidence = 0.5):
        self.mode = mode
        self.upperBody = upperBody
        self.smooth = smooth
        self.model_complexity = model_complexity
        self.detection_confidence = detection_confidence
        self.track_confidence = track_confidence

        # drawing pose landmarks
        self.mpDraw = mp.solutions.drawing_utils
        # creating mediapipe pose objects
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.model_complexity, self.smooth, self.upperBody,
                                     self.detection_confidence, self.track_confidence)

    def findPose(self, img, draw=True, landmark_size=10, landmark_color=(255, 0, 0)):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(img_rgb)
        # checking if any pose is detected
        if self.results.pose_landmarks:
            # getting the index and position of each landmark
            for index, landmark in enumerate(self.results.pose_landmarks.landmark):
                # converting x, y coordinates (given as aspect ratios of the image) to pixels
                height, width, channel = img.shape
                center_x, center_y = int(landmark.x * width), int(landmark.y * height)
                cv2.circle(img, (center_x, center_y), landmark_size, landmark_color, cv2.FILLED)
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img
    
    def findPosition(self, img, draw=True, landmark_size=10, landmark_color=(255, 0, 0)):
        landmark_list = []
        # checking if any pose is detected
        if self.results.pose_landmarks:
        # getting the index and position of each landmark
            for index, landmark in enumerate(self.results.pose_landmarks.landmark):
                # converting x, y coordinates (given as aspect ratios of the image) to pixels
                height, width, channel = img.shape
                center_x, center_y = int(landmark.x * width), int(landmark.y * height)
                landmark_list.append([index, center_x, center_y])
                if draw:
                    cv2.circle(img, (center_x, center_y), landmark_size, landmark_color, cv2.FILLED)
        return landmark_list

def main():
    cap = cv2.VideoCapture("PoseEstimationProject/videos/defante.mp4")
   
    detector = poseDetector()
    while True:
        sucess, img = cap.read()
        detector.findPose(img, landmark_size=10, landmark_color=(0, 0, 255))
        #landmark_list = detector.findPosition(img, landmark_size, landmark_color)
        #print(landmark_list[14])
        img = cv2.resize(img, (300, 620)) 
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()