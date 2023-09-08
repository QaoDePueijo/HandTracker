import cv2
import mediapipe as mp
import time
import handTrack as htm


#past Time
pTime = 0
#current Time
cTime = 0

cap = cv2.VideoCapture(0) #change camera
detector = htm.handDetector()

"""
This code snippet is a while loop that continuously reads frames from a video capture device, detects hands in the frames using a hand detection model, and performs some operations on the detected hands. It also calculates and displays the frames per second (FPS) on the image.

Inputs:
- cap: A video capture object created using cv2.VideoCapture(0).
- detector: An instance of the handDetector class from the handTrack module.

Outputs:
- An annotated image with hand landmarks and FPS displayed.
"""

while True:
    success, img = cap.read()
    if not success:
        break
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    if len(lmList) != 0:
        print(lmList[4])
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 25, 255), 1)
    cv2.imshow("Image", img)
    if cv2.waitKey(1) == ord('q'):
        break