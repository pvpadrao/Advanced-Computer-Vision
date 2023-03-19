import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

# creating a mediapipe object
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=10)

# drawing hand landmarks
mpDraw = mp.solutions.drawing_utils
drawing_specs = mpDraw.DrawingSpec(color=(0, 255, 0),thickness=2, circle_radius=2)

# setting time
previous_time = 0

while True:
    sucess, img = cap.read()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    # checking if hands are detected
    #print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # getting the index and position of each landmark
            for index, landmark in enumerate(hand_landmarks.landmark):
                #print(index, landmark)
                # converting x, y coordinates (given as aspect ratios of the image) to pixels
                height, width, channel = img.shape
                center_x, center_y = int(landmark.x * width), int(landmark.y * height)
                #print(index, center_x, center_y)

            mpDraw.draw_landmarks(img, hand_landmarks, mpHands.HAND_CONNECTIONS, drawing_specs)

    # calculting fps and printing to window
    current_time = time.time()
    fps = 1 / (current_time - previous_time)
    previous_time = current_time
    cv2.putText(img, f'fps: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
