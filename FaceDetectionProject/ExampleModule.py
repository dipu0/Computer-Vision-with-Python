import cv2
import mediapipe as mp
import time
import FaceDetectionModule as fd

prevTime = 0
cap = cv2.VideoCapture(0)

detector = fd.FaceDetector()

while True:
    success, img = cap.read()
    img, boundingboxes = detector.findFaces(img)
    # print(boundingboxes)

    currTime = time.time()
    fps = 1 / (currTime - prevTime)
    prevTime = currTime

    cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 3)
    cv2.imshow("Frame", img)

    if cv2.waitKey(1) == ord('q'):
        break