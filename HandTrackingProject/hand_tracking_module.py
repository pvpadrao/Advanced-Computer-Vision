import cv2
import mediapipe as mp
import time

class handDetector():
    def __init__(self, mode=False, maxHands = 2, model_complexity = 1, detection_confidence = 0.5, 
                 track_confidence = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.model_complexity = model_complexity
        self.detection_confidence = detection_confidence
        self.track_confidence = track_confidence
        # creating a mediapipe object
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.model_complexity, 
                                        self.detection_confidence, self.track_confidence)
        # drawing hand landmarks
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)
        # checking if hands are detected
        #print(results.multi_hand_landmarks)
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, hand_landmarks, self.mpHands.HAND_CONNECTIONS)
        return img
    
    def findHandLandmarkPosition(self, img, hand_number = 0, draw=False):
        landmark_list = []
        # checking if hands are detected
        if self.results.multi_hand_landmarks:
                hand = self.results.multi_hand_landmarks[hand_number]
            # getting the index and position of each landmark
                for index, landmark in enumerate(hand.landmark):
                    #converting x, y coordinates (given as aspect ratios of the image) to pixels
                    height, width, channel = img.shape
                    center_x, center_y = int(landmark.x * width), int(landmark.y * height)
                    landmark_list.append([index, center_x, center_y])
                    if draw:
                        cv2.circle(img, (center_x, center_y), 10, (255, 0, 255), cv2.FILLED)

        return landmark_list
                

def main():
    cap = cv2.VideoCapture(0)
    # setting time
    previous_time = 0
    current_time = 0
    detector = handDetector()
    while True:
        sucess, img = cap.read()
        img = detector.findHands(img)
        landmark_list = detector.findHandLandmarkPosition(img)
        # checking if hand is detected
        if len(landmark_list) !=0:
            print(landmark_list[4])
        #calculting fps and printing to window
        current_time = time.time()
        fps = 1 / (current_time - previous_time)
        previous_time = current_time
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 0), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(1)
 
if __name__ == "__main__":
    main()