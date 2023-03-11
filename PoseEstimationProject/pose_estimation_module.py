import cv2
import mediapipe as mp


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

    def findPose(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(img_rgb)
        # checking if any pose is detected
        if results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img

def main():
    cap = cv2.VideoCapture("PoseEstimationProject/videos/defante.mp4")
    detector = poseDetector()
    while True:
        sucess, img = cap.read()
        detector.findPose(img)
        img = cv2.resize(img, (300, 620)) 
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()