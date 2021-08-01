
import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('C:/opencv-master/opencv-master/data/haarcascades/haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
id = input('Enter User ID : ')
sampleN=0;
while 1:
    ret, img = cap.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5);
    for (x,y,w,h) in faces:
        sampleN=sampleN+1;
        cv2.imwrite("dataSet/user."+str(id)+                   "." +str(sampleN)+ ".jpg", gray[y:y+h, x:x+w])
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        cv2.waitKey(1);
    cv2.imshow('img',img);
    cv2.waitKey(300);
    if sampleN > 100:
        break;

cap.release()
cv2.destroyAllWindows()