
import cv2
import mediapipe as mp
import time

class handDetector():
    def __init__(self, mode = False,
                maxHands = 2,modelComplex = 0,
                detection_confidence = 0.5,
                tracking_confidence=0.5):
        self.mode = mode
        self.modelComplex = modelComplex
        self.maxHands = maxHands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode,self.maxHands,self.modelComplex,self.detection_confidence,self.tracking_confidence)
        self.mpDraw = mp.solutions.drawing_utils
    
    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        #print(results.multi_hand_landmarks)
        
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img,handLms, self.mpHands.HAND_CONNECTIONS)
                
        return img

    def findPosition(self,img,handNo=0,draw=True):

        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]

            for id, lm in enumerate(myHand.landmark):
                #print(id, lm)
                h, w, c = img.shape
                cx,cy = int(lm.x*w),int(lm.y*h)
                lmList.append([id,cx,cy])
                if draw:
                    cv2.circle(img,(cx,cy),7,(0,255,255),cv2.FILLED)
                #print(id, cx, cy)
        
        return lmList

        

def main():
    #past Time
    pTime = 0
    #current Time
    cTime = 0
    
    cap = cv2.VideoCapture(1) #change camera
    detector = handDetector()

    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img, draw=True)
        if len(lmList) !=0:
            print(lmList[4])
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,(255,25,255),1)


        cv2.imshow("Image",img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()