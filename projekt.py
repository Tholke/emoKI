# -*- coding: utf-8 -*-
"""
Created on Sun Jan  8 15:54:15 2017

@author: Thomas
"""

import cv2
import numpy as np

img = cv2.imread('i1.jpg')
imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face = face_cascade.detectMultiScale(imggray, 1.1, 5)


for(x,y,w,h) in face:
    cv2.rectangle(img, (x-2,y-2), (x+w+2, y+h+2), (230, 10, 100), 2)
    faceimg = img[y:y+h, x:x+w]
    
    

    

cv2.imshow('IMAGE', img)
cv2.imshow('FACEIMAGE', faceimg)
cv2.waitKey(0)
cv2.destroyAllWindows()
