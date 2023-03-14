import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

# creating a mediapipe object
mpFaceMesh = mp.solutions.face_mesh
'''
Default Input Args:
static_image_mode=False,
max_num_faces=1,
refine_landmarks=False,
min_detection_confidence=0.5,
min_tracking_confidence=0.5
'''
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)

# drawing face landmarks
mpDraw = mp.solutions.drawing_utils
drawing_specs = mpDraw.DrawingSpec(color=(255, 0, 0),thickness=1, circle_radius=1)

previous_time = 0

while True:
    sucess, img = cap.read()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(img_rgb)

    if results.multi_face_landmarks:
            # getting the index and position of each detected face
            for index, detected_face_landmarks in enumerate(results.multi_face_landmarks):
                mpDraw.draw_landmarks(img, detected_face_landmarks, mpFaceMesh.FACEMESH_CONTOURS, drawing_specs)
                # getting the position (in pixels) of each face landmark
                for index, landmarks in enumerate(detected_face_landmarks.landmark):
                    # converting x, y coordinates to pixels
                    # getting shape of the image
                    height, width, channel = img.shape
                    # converting x, y of the relative bounding box to pixels
                    x, y = int(landmarks.x * width), int(landmarks.y * height)
                    print(index, x, y)

    #calculting fps and printing to window
    current_time = time.time()
    fps = 1 / (current_time - previous_time)
    previous_time = current_time
    img = cv2.flip(img,1)
    cv2.putText(img, f'fps: {int(fps)}', (10, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
