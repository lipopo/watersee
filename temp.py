# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import cv2
import numpy as np

def getdeg(a):
    if a[0]*a[1] > 0:
        a[2] = np.arctan(a[1]/a[0])
        if a[0] < 0:
            a[2] = a[2] + np.pi
    if a[0]*a[1] < 0:
        a[2] = np.arctan(a[1]/a[0]) + 2*np.pi
        if a[0] < 0:
            a[2] = a[2] - np.pi
    if a[0]*a[1] == 0:
        a[2] = 0
        if a[0] < 0:
            a[2] = np.pi
        if a[1] > 0:
            a[2] = np.pi/2
        if dushu[i][1] < 0:
            a[2] = 3*np.pi/2
    return a
    

img = cv2.imread("C:/Users/zhangyating/Documents/lipo_res/shuibiao/use.png")

img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

cirs = cv2.HoughCircles(img_gray,cv2.cv.CV_HOUGH_GRADIENT,1,30,minRadius=20,maxRadius=50)

if cirs.shape[1]<8:
    print "No Circle In Here!"
    exit()

if cirs.shape[1]>8:
    
#確定圓心
leijiaqi = np.zeros(img_gray.shape,dtype=np.float32)    
for i in range(0,8):
    for j in range(0,8):
        x0 = cirs[0,i,0]
        y0 = cirs[0,i,0]
        x1 = cirs[0,j,0]
        y1 = cirs[0,j,1]
        xz = (x0+x1)/2
        yz = (y0+y1)/2
        xmax = img_gray.shape[1]
        zeros = np.zeros(img_gray.shape,dtype=np.float32)        
        if y0 != y1:            
            k = -(x0-x1)/(y0-y1)
            cv2.line(zeros,(0,int(k*(0-xz)+yz)),(xmax,int(k*(xmax-xz)+yz)),1)
        if y0 == y1:
            cv2.line(zeros,(0,y0),(xmax,y0),1)
        leijiaqi = leijiaqi +zeros

if leijiaqi.max() < 8:
    print "Error Center of Biaopan"
    exit()
x0,y0 = leijiaqi(np.where(leijiaqi > leijiaqi.max() - 0.1))
x0 = np.average(x0)
y0 = np.average(y0)

#確定各點度數
dushu = np.zeros(8,dtype=[('x',np.float32),('y',np.float32),('tag',np.float32),('r',np.float32)])
for i in range(0,8):
    dushu[i][0] = cirs[0,i,0] - x0
    dushu[i][1] = cirs[0,i,1] - y0
    dushu[i][3] = cirs[0,i,2]
    dushu[i] = getdeg(dushu[i])

#從x軸開始，順時針排列各點  確定相鄰兩點之間的度數lines
dushu_use = np.sort(dushu,order='tag')
lines = np.zeros(8,dtype=np.float32)
lines[0] = dushu_use[0][2] + (2*np.pi-dushu_use[-1][2]) 
for i in range(1,8):
    lines[i] = dushu_use[i][2]-dushu_use[i-1][2]
    
rotateS = np.where(lines > lines.max() - 0.01)[0]
rotateX = rotateS-1

#rotateX為逆時針起點，以此建立逆時針方向的矩陣
dingxiang = np.zeros(8,dtype=[('x',np.float32),('y',np.float32),('r',np.float32),('biaoliang',np.float32)])
for i in range(0,8):
    dingxiang[i][0] = dushu_use[rotateX-i][0]+x0
    dingxiang[i][1] = dushu_use[rotateX-i][1]+y0
    dingxiang[i][2] = dushu_use[rotateX-i][3]
    dingxiang[i][3] = 1000/(10**i)

#求取每個子錶盤的度數
#得到只含有子錶盤的圖像

#圖像開運算 并進行霍夫曼直線檢測
cv2.morphologyEx(img,cv2.MORPH_OPEN,np.ones((5,5),dtype=np.uint8)
lines = cv2.HoughLinesP()

#依次求出直線度數

#依據正方向確定指針讀數

#擺正錶盤
(x,y) = ((dushu_use[rotateS][0]+dushu_use[rotateX][0])/2,(dushu_use[rotateS][1]+dushu_use[rotateX][1])/2)
rotate = 0
(x,y,rotate) = getdeg((x,y,rotate))

rotMat = cv2.getRotationMatrix2D((x0,y0),180*(rotate+np.pi/2)/np.pi,1)
img_out = cv2.warpAffine(img,rotMat,img.shape)





    
    