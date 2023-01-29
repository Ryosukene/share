#!/usr/bin/env python
# coding: utf-8

# In[ ]:

'''
import cv2

def onMouse(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)

img = cv2.imread('George_electrodes.png')
cv2.imshow('sample', img)
cv2.setMouseCallback('sample', onMouse)
cv2.waitKey(0)


# In[ ]:

'''
import cv2
import numpy as np
file_name='George_electrodes.png'
img=cv2.imread(file_name,cv2.IMREAD_GRAYSCALE)

def click_pos(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        img2 = np.copy(img)
        cv2.circle(img2,center=(x,y),radius=5,color=255,thickness=-1)
        pos_str='(x,y)=('+str(x)+','+str(y)+')'
        cv2.putText(img2,pos_str,(x+10, y+10),cv2.FONT_HERSHEY_PLAIN,2,255,2,cv2.LINE_AA)
        cv2.imshow('window', img2)
        
cv2.imshow('window', img)
cv2.setMouseCallback('window', click_pos)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:




