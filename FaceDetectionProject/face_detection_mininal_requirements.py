import cv2
import mediapipe as mp
import time

# creating a mediapipe object
mpFaceDetection = mp.solutions.face_detection
# min_detection_confidence is a threshold that can be changed
faces = mpFaceDetection.FaceDetection(min_detection_confidence = 0.5)

# drawing hand landmarks
mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# setting time for fps calculation 
previous_time = 0

while True:
    sucess, img = cap.read()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faces.process(img_rgb)

    if results.detections:
            # getting the index and position of each detected face
            for index, detected_face in enumerate(results.detections):
                # getting the bounding box relative to the class for each detected face
                bounding_box_relative = detected_face.location_data.relative_bounding_box
                # getting shape of the image
                height, width, channel = img.shape
                # converting x, y of the relative bounding box to pixels
                bounding_box = int(bounding_box_relative.xmin * width), int(bounding_box_relative.ymin * height), \
                               int(bounding_box_relative.width * width), int(bounding_box_relative.height * height)
                cv2.rectangle(img, bounding_box, (0, 0, 255), 2)
                cv2.putText(img, f'{int(detected_face.score[0]*100)}%', (bounding_box[0], bounding_box[1]-20), 
                              cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)

    # calculating fps and output to window
    current_time = time.time()
    fps = 1 / (current_time - previous_time)
    previous_time = current_time
    img = cv2.flip(img,1)
    cv2.putText(img, f'fps: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)

    cv2.imshow("Image", img)
    cv2.waitKey(1)