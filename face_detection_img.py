#!/usr/bin/env python
# coding: utf-8

# In[119]:


#import library
import cv2
import matplotlib.pyplot as plt


# In[120]:


#read face imgs
Sara = cv2.imread("Sara.jpg")
Emma = cv2.imread("Emma.jpg")


# In[121]:


#build cascade obg
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")


# In[122]:


#define detection function
def detection(img):
    
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    face_cord = face_cascade.detectMultiScale(gray_img,1.3,8)
    eys_cord = eye_cascade.detectMultiScale(gray_img,1.3,8)
    
    for x, y, w, h in eys_cord:
        cv2.rectangle(gray_img, (x, y), (x+w, y+h), (255,0,0),20)
            
    for x, y, w, h in face_cord:
        cv2.rectangle(gray_img, (x, y), (x+w, y+h), (255,0,0),20) 
        
    return gray_img


# In[123]:


#Sara_detection
detected_img = detection(Sara)
plt.imshow(detected_img,cmap = 'gray')


# In[124]:


#Emma_detection
detected_img = detection(Emma)
plt.imshow(detected_img,cmap = 'gray')


# In[ ]:





# In[ ]:




