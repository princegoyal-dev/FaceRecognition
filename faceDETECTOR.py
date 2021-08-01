
# -*- coding: utf-8 -*-
"""
Created on Mon May 28 10:55:22 2018

@author: Admin
"""

import numpy as np
import cv2
import os

fname = "Trainning Data/trainingdata.yml"
if not os.path.isfile(fname):
  print("Please train the data first")
  exit(0)
#print(fname)  
face_cascade = cv2.CascadeClassifier('C:/opencv-master/opencv-master/data/haarcascades/haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(fname)
print(recognizer.read(fname))
id=0
#str=""
#font=cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_COMPLEX_SMALL,5,1,0,4)
while True:
    ret, img = cap.read()
    #gray = img
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5);
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        id,conf=recognizer.predict(gray[y:y+h,x:x+w])
        print("ID : ",id)
        if id==1:
            id="prince"
        elif id==2:
            id="himanshu"
        elif id==3:
            id="shivam"
        elif id==4:
            id="Kaushik"
        elif id==5:
            id="Chirag"   
        elif id == 6:
            id = "varun"
        else:
            id="No Match Found"  
        cv2.putText(img, str(id), (x+2,y+h-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)
    cv2.imshow('Face Recognizer',img)
    if cv2.waitKey(1) == ord('q'):
        break;
cap.release()
cv2.destroyAllWindows()