# Modules used
import cv2
import mediapipe as mp
import time

class handTrack():
    def __init__(self,static_image_mode=False,
                 max_number_of_hands=2,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        
        self.static_image_mode = static_image_mode
        self.max_number_of_hands = max_number_of_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.draw = mp.solutions.drawing_utils
        self.hands = mp.solutions.hands
        self.hand =  self.hands.Hands(self.static_image_mode,
                                      self.max_number_of_hands,
                                      self.min_detection_confidence,
                                      self.min_tracking_confidence)
       
    def find_number_of_hands(self,frame,to_draw=True):
        img = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        self.results = self.hand.process(img)
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if to_draw:
                    # Drawing the landmarks and connecting them 
                    self.draw.draw_landmarks(frame,hand_landmarks,self.hands.HAND_CONNECTIONS)
        return frame
    
    def find_position_of_hands(self,frame,hand_number=0):
        lmlist = []
        if self.results.multi_hand_landmarks:
            hand_landmark = self.results.multi_hand_landmarks[hand_number]
            for id,lm in enumerate(hand_landmark.landmark):
                w,h,c = frame.shape
                # Position of coordinates 
                cx = int(lm.x*h)
                cy = int(lm.y*w)
                lmlist.append([id,cx,cy])
                if to_draw:
                    if id == 8:
                        cv2.circle(frame,(cx,cy),20,(255,0,0),-1)
        return lmlist
        
def main():
    ct,pt = 0,0
    # Live video Capture
    cap = cv2.VideoCapture(0) 
    cap.set(3,1080)
    cap.set(4,720)
    
    detect = handTrack()
    # Looping
    while cap.isOpened:
        ret,frame = cap.read()
        if not ret:
            continue
        # Flip the live capture
        frame = cv2.flip(frame,1)
        
        frame = detect.find_number_of_hands(frame)
        position = detect.find_position_of_hands(frame)
        #if len(position) != 0:
        #   print(position)
        
        ct = time.time()
        fps = 1 / (ct - pt)
        pt = ct
        cv2.putText(frame,str(int(fps)),(30,120),cv2.FONT_HERSHEY_PLAIN,3,(225,180,255),3)
        
        cv2.imshow("frame",frame)
        cv2.waitKey(1)
    
if __name__ == "__main__":
    main()
