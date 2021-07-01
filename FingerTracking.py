import cv2
import time 
import Hand_Tracking as ht

detect = ht.handTrack()
ct,pt = 0,0
cap = cv2.VideoCapture(0)
cap.set(3,1080)
cap.set(4,720)

fingers = []
points = [8,12,16,20]
x = 0

while True:
    ret,frame = cap.read()
    if not ret:
        continue
    frame = cv2.flip(frame,1)
    
    frame = detect.find_number_of_hands(frame)
    position = detect.find_position_of_hands(frame,to_draw=False)    
    if len(position) != 0:
        if position[4][1] < position[3][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        
        for i in points:
            if position[i][2] < position[i-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        
        x = x + sum(fingers)
        cv2.putText(frame,f"Number of fingers = {str(x)})",(30,600),cv2.FONT_HERSHEY_PLAIN,5,(255,0,0),5)
        x = x - sum(fingers)
        fingers.clear()
        
        ct = time.time()
        fps = 1/(ct-pt)
        pt = ct
        cv2.putText(frame,str(int(fps)),(30,60),cv2.FONT_HERSHEY_PLAIN,3,(255,255,0),3)
    
        cv2.imshow("frame",frame)
        if cv2.waitKey(1) & 0xff == ord("q"):
            break
    
cap.release()
cv2.destroyAllWindows()
