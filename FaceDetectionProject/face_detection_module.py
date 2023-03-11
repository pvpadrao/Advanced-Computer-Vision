import cv2
import mediapipe as mp
import time
class faceDetector():
    def __init__(self, model_selection=1, mode=False, detection_confidence = 0.5):
        self.model_selection = model_selection
        self.mode = mode
        self.detection_confidence = detection_confidence
        # drawing pose landmarks
        self.mpDraw = mp.solutions.drawing_utils
        # creating mediapipe pose objects
        # creating a mediapipe object
        self.mpFaceDetection = mp.solutions.face_detection
        # min_detection_confidence is a threshold that can be changed
        self.faces = self.mpFaceDetection.FaceDetection(self.detection_confidence)


    def findFace(self, img, draw=True, bounding_box_thickness=2, bounding_box_color=(0, 0, 255)):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faces.process(img_rgb)
        bounding_boxes_list = []
        # checking if any pose is detected
        if self.results.detections:
            # getting the index and position of each detected face
            for index, detected_face in enumerate(self.results.detections):
                # getting the bounding box relative to the class for each detected face
                bounding_box_relative = detected_face.location_data.relative_bounding_box
                # getting shape of the image
                height, width, channel = img.shape
                # converting x, y of the relative bounding box to pixels
                bounding_box = int(bounding_box_relative.xmin * width), int(bounding_box_relative.ymin * height), \
                               int(bounding_box_relative.width * width), int(bounding_box_relative.height * height)
                bounding_boxes_list.append([bounding_box, detected_face.score])
                cv2.rectangle(img, bounding_box, bounding_box_color, bounding_box_thickness)
                cv2.putText(img, f'{int(detected_face.score[0]*100)}%', (bounding_box[0], bounding_box[1]-20), 
                                cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
        return img, bounding_boxes_list

def main():
    cap = cv2.VideoCapture(0)
    previous_time = 0
    detector = faceDetector()
    while True:
        sucess, img = cap.read()
        img, bounding_boxes_list = detector.findFace(img, bounding_box_thickness=1, bounding_box_color=(255, 0, 255))
        # calculating fps and output to window
        current_time = time.time()
        fps = 1 / (current_time - previous_time)
        previous_time = current_time
        cv2.putText(img, f'fps: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()