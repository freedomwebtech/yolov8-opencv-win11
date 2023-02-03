import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO



model=YOLO('yolov8s.pt')


def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)
        

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap=cv2.VideoCapture('vidyolov8.mp4')

my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")
print(class_list)
count=0

area=[(261,207),(256,251),(668,196),(644,159)]
while True:
    
    ret,frame = cap.read()
    frame=cv2.resize(frame,(1020,500))
    if ret is None:
        break
    count += 1
    if count % 3 != 0:
        continue

    results=model.predict(frame)
    a=results[0].boxes.boxes
#    print(a)
    px = pd.DataFrame(a).astype("float")
#    print(px)
    for index,row in px.iterrows():
#        print(row)
         x1=int(row[0])
         y1=int(row[1])
         x2=int(row[2])
         y2=int(row[3])
         d=int(row[5])
         c=class_list[d]
 #        if 'car' in c:
         cx=int(x1+x2)//2
         cy=int(y1+y2)//2
#             results=cv2.pointPolygonTest(np.array(area,np.int32),((cx,cy)),False)
#             if results>=0:
         cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)
         cv2.circle(frame,(cx,cy),4,(0,255,0),-1)
         cv2.putText(frame,str(c),(x1,y1),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,0,0),1)
 #   cv2.polylines(frame,[np.array(area,np.int32)],True,(255,0,0),2)              
    cv2.imshow("RGB", frame)
    if cv2.waitKey(1)&0xFF==27:
        break

cap.release()
cv2.destroyAllWindows()
