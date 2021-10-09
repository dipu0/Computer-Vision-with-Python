import cv2
import mediapipe as mp
import time

from mediapipe.python.solutions import face_detection

class FaceDetector():
    def __init__(self, minDetectionCon=0.5):
        self.minDetectionCon = minDetectionCon
        self.mpfaceDetectn = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.facedetection = self.mpfaceDetectn.FaceDetection(self.minDetectionCon)

    def findFaces(self, img, draw=True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.facedetection.process(imgRGB)
        # print(self.results)
        boundingboxes = []
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                boundingboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                boundingbox = int(boundingboxC.xmin * iw), int(boundingboxC.ymin * ih), \
                              int(boundingboxC.width * iw), int(boundingboxC.height * ih)
                boundingboxes.append([id, boundingbox, detection.score])

                if draw:
                    img = self.fancyDraw(img, boundingbox)
                    cv2.putText(img, f'{int(detection.score[0] * 100)}%', (boundingbox[0], boundingbox[1] - 20),
                                cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

        return img, boundingboxes

    def fancyDraw(self, img, boundingbox, l=30, t=5, rt=1):
        x, y, w, h = boundingbox
        x1, y1 = x + w, y + h

        cv2.rectangle(img, boundingbox, (255, 0, 255), rt)

        # top left : x,y
        cv2.line(img, (x, y), (x + l, y), (255, 0, 255), t)
        cv2.line(img, (x, y), (x, y + l), (255, 0, 255), t)

        # top right : x1,y
        cv2.line(img, (x1, y), (x1 - l, y), (255, 0, 255), t)
        cv2.line(img, (x1, y), (x1, y + l), (255, 0, 255), t)

        # bottom left : x,y1
        cv2.line(img, (x, y1), (x + l, y1), (255, 0, 255), t)
        cv2.line(img, (x, y1), (x, y1 - l), (255, 0, 255), t)

        # bottom right : x1,y1
        cv2.line(img, (x1, y1), (x1 - l, y1), (255, 0, 255), t)
        cv2.line(img, (x1, y1), (x1, y1 - l), (255, 0, 255), t)

        return img


def main():
    cap = cv2.VideoCapture(0)
    wCam, hCam = 1280, 720
    cap.set(3, wCam)
    cap.set(4, hCam)

    prevTime = 0

    detector = FaceDetector()

    while True:
        success, img = cap.read()
        img, boundingboxes = detector.findFaces(img)
        # print(boundingboxes)

        currTime = time.time()
        fps = 1 / (currTime - prevTime)
        prevTime = currTime

        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3)
        cv2.imshow("Frame", img)

        if cv2.waitKey(1) == ord('q'):
            break


if __name__ == "__main__":
    main()