import cv2
import mediapipe as mp
import time

# creating mediapipe pose objects
mpPose = mp.solutions.pose
pose = mpPose.Pose()

# drawing pose landmarks
mpDraw = mp.solutions.drawing_utils

# reading videos
#cap = cv2.VideoCapture("PoseEstimationProject/videos/defante.mp4")
cap = cv2.VideoCapture(0)
previous_time = 0
while True:
    sucess, img = cap.read()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    if results.pose_landmarks:
        # getting the index and position of each landmark
        for index, landmark in enumerate(results.pose_landmarks.landmark):
            # converting x, y coordinates (given as aspect ratios of the image) to pixels
            height, width, channel = img.shape
            center_x, center_y = int(landmark.x * width), int(landmark.y * height)
            cv2.circle(img, (center_x, center_y), 8, (255, 0, 255), cv2.FILLED)
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
    # calculting fps and printing to window
    current_time = time.time()
    fps = 1 / (current_time - previous_time)
    previous_time = current_time
    #img = cv2.resize(img, (300, 620))
    cv2.putText(img, f'fps: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
    cv2.imshow("Image", img)
    cv2.waitKey(1)


    

    