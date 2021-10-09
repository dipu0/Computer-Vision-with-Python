import cv2
import time
import PoseModule as pm

cap = cv2.VideoCapture(0)
pTime = 0
detector = pm.poseDetection()
while True:
    success, img = cap.read()
    img = detector.findPose(img)
    lmList = detector.findPostion(img)
    print(lmList)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
    cv2.imshow("pose detection", img)

    if cv2.waitKey(40) == ord("q"):
        break

