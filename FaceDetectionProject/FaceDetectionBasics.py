import cv2
import mediapipe as mp
import time

from mediapipe.python.solutions import face_detection

cap = cv2.VideoCapture(0)
wCam , hCam = 1280 , 720
cap.set(3 , wCam)
cap.set(4 , hCam)

prevTime =0

mpfaceDetectn = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
facedetection =mpfaceDetectn.FaceDetection(0.8)


while True:
    success , img = cap.read()

    imgRGB = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
    results = facedetection.process(imgRGB)
    # print(results)

    if results.detections:
        for id,detection in enumerate(results.detections):
            # mpDraw.draw_detection(img , detection)
            # print(id,detection)
            # print(detection.score)
            # print(detection.location_data.relative_bounding_box)
            boundingboxC = detection.location_data.relative_bounding_box
            ih , iw , ic = img.shape
            boundingbox = int(boundingboxC.xmin*iw) , int(boundingboxC.ymin*ih) , \
                          int(boundingboxC.width*iw) , int(boundingboxC.height*ih)
            cv2.rectangle(img , boundingbox , (255, 0 ,255) , 2)
            cv2.putText(img , f'{int(detection.score[0]*100)}%', (boundingbox[0],boundingbox[1]-20), cv2.FONT_HERSHEY_PLAIN, 1.5 , (0,255,0),2)

    currTime  = time.time()
    fps = 1/(currTime - prevTime)
    prevTime = currTime

    cv2.putText(img , f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_PLAIN, 2 , (255,0,0),3)
    cv2.imshow("Frame" , img)

    if cv2.waitKey(1) == ord('q'):
        break