import cv2

cap = cv2.VideoCapture(0)


while True:
    sucess, img = cap.read()

    cv2.imshow("Image", img)
    cv2.waitKey(1)