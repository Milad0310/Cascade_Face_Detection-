#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#import library
import cv2


# In[ ]:


#build cascade obj 
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")


# In[ ]:


#define detection function

def detection(gray, frame):
    
    face = face_cascade.detectMultiScale(gray, 1.3, 5)
    for x, y, w, h in face:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (100, 50, 200), 3)

    eye = eye_cascade.detectMultiScale(gray, 1.3, 3)
    for ex, ey, ew, eh in eye:
        cv2.rectangle(frame, (ex,ey), (ex+ew, ey+eh), (130, 250, 20), 3)
    
    return frame


# In[ ]:


#Capture video from webcam

cap_vid = cv2.VideoCapture(0)

while True:
    
    _, frame = cap_vid.read() 
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    detected_fram = detection(gray, frame)
    
    cv2.imshow('detected_face',detected_fram)
    
    if cv2.waitKey(0) & 0xff == ord("q"):
        break
        
cap_vid.release()
cv2.destroyAllWindows()       


# In[ ]:




