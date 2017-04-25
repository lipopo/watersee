

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

def save(img,i):
    return cv2.imwrite("C:/Users/Administrator/Desktop/imgs/0.0.1/"+str(i)+".png",img)

img = cv2.imread("C:/Users/Administrator/Desktop/imgs/use.png")

img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
save(img_gray,2)

#cirs = np.array([],dtype=[('x',np.float32),('y',np.float32),('r',np.float32)])
cirs = cv2.HoughCircles(img_gray,cv2.cv.CV_HOUGH_GRADIENT,1,30,maxRadius=50)

if cirs.shape[1]<8:
    print "No Circle In Here!"
    exit()

#if cirs.shape[1]>8:

#圆调试
img_cir = np.copy(img)
for i in range(0,cirs.shape[1]):
    cv2.circle(img_cir,(cirs[0,i,0],cirs[0,i,1]),cirs[0,i,2],np.random.randint(255,size=3))    
save(img_cir,3)
    
#確定圓心
leijiaqi = np.zeros(img_gray.shape,dtype=np.float32)    
for i in range(0,8):
    for j in range(0,8):
        if i == j:
            break
        x0 = cirs[0,i,0]
        y0 = cirs[0,i,1]
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

img_line = np.copy(img)
for i in range(0,8):
    for j in range(0,8):
        if i == j:
            break
        x0 = cirs[0,i,0]
        y0 = cirs[0,i,1]
        x1 = cirs[0,j,0]
        y1 = cirs[0,j,1]
        cv2.line(img_line,(x0,y0),(x1,y1),np.random.randint(255,size=3))
        xz = (x0+x1)/2
        yz = (y0+y1)/2
        cv2.circle(img_line,(int(xz),int(yz)),2,np.random.randint(255,size=3),-1)
        xmax = img_gray.shape[1]        
        if y0 != y1:            
            k = -(x0-x1)/(y0-y1)
            cv2.line(img_line,(0,int(k*(0-xz)+yz)),(xmax,int(k*(xmax-xz)+yz)),np.random.randint(255,size=3))
        if y0 == y1:
            cv2.line(img_line,(0,y0),(xmax,y0),np.random.randint(255,size=3))       
save(img_line,4)

if leijiaqi.max() < 8:
    print "Error Center of Biaopan"
    exit()
xzzz,yzzz = np.where(leijiaqi > leijiaqi.max() - 0.1)
x0 = np.average(yzzz)
y0 = np.average(xzzz)

img_line2 = np.copy(img)
for i in range(0,8):
    cv2.line(img_line2,(int(x0),int(y0)),(cirs[0,i,0],cirs[0,i,1]),np.random.randint(255,size=3))
save(img_line2,5)

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
dingxiang = np.zeros(8,dtype=[('x',np.float32),('y',np.float32),('r',np.float32),('biaoliang',np.float64)])
for i in range(0,8):
    dingxiang[i][0] = np.float32(dushu_use[rotateX-i]['x']) + x0
    dingxiang[i][1] = np.float32(dushu_use[rotateX-i]['y']) + y0
    dingxiang[i][2] = dushu_use[rotateX-i]['r']
    dingxiang[i][3] = 0.0001*(10**i)
    
img_text = np.copy(img)
for i in range(0,8):
    cv2.putText(img_text,str(i),(dingxiang[i][0],dingxiang[i][1]),cv2.FONT_HERSHEY_SIMPLEX,1,np.random.randint(255,size=3),2)
save(img_text,6)

#求取每個子錶盤的度數
#得到只含有子錶盤的圖像
zeros = np.zeros(img_gray.shape,dtype=np.uint8)
for i in range(0,8):
    x_0 = dingxiang[i][0]
    y_0 = dingxiang[i][1]
    r = dingxiang[i][2]
    cv2.circle(zeros,(x_0,y_0),r,255,-1)

img_zibiaopan = cv2.bitwise_and(img_gray,img_gray,mask=zeros)
save(img_zibiaopan,7)


#擺正錶盤
(x,y) = ((np.float32(dushu_use[rotateS]['x'])+np.float32(dushu_use[rotateX]['x']))/2,(np.float32(dushu_use[rotateS]['y'])+np.float32(dushu_use[rotateX]['y']))/2)
rotate = 0
answer = getdeg([x,y,rotate])
x = answer[0]
y = answer[1]
rotate = answer[2]
img_33 = np.copy(img)

[a,b,rotate] = getdeg([((dingxiang[0]['x']+dingxiang[-1]['x'])/2)-x0,((dingxiang[0]['y']+dingxiang[-1]['y'])/2)-y0,0])

cv2.circle(img_33,(int(x0),int(y0)),2,(0,0,255),-1)
cv2.circle(img_33,(dingxiang[0]['x'],dingxiang[0]['y']),2,(0,0,255),-1)
cv2.circle(img_33,(dingxiang[-1]['x'],dingxiang[-1]['y']),2,(0,0,255),-1)
cv2.circle(img_33,(int((dingxiang[0]['x']+dingxiang[-1]['x'])/2),int((dingxiang[0]['y']+dingxiang[-1]['y'])/2)),2,(0,0,255),-1)
cv2.putText(img_33,str(rotate),(int((dingxiang[0]['x']+dingxiang[-1]['x'])/2),int((dingxiang[0]['y']+dingxiang[-1]['y'])/2)),cv2.FONT_HERSHEY_SIMPLEX,1,np.random.randint(255,size=3),2)
cv2.line(img_33,(int(x0),int(y0)),(int((dingxiang[0]['x']+dingxiang[-1]['x'])/2),int((dingxiang[0]['y']+dingxiang[-1]['y'])/2)),(0,255,0),1)
save(img_33,152)


rotMat = cv2.getRotationMatrix2D((x0,y0),180*(rotate-np.pi/2)/np.pi,1)
img_out = cv2.warpAffine(img,rotMat,img_gray.shape)
save(img_out,11)

#二值化 圖像開運算 并進行霍夫曼直線檢測
mask = img_zibiaopan < 90
mask.dtype = img_zibiaopan.dtype
img_otu = cv2.bitwise_and(mask * 255,mask * 255,mask=zeros)
save(img_otu,8)
img_h = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)[:,:,0]
img_mask = img_h < 10
img_mask.dtype = np.uint8
img_mask = img_mask * 255
img_otum = cv2.bitwise_and(img_otu,img_otu,mask = img_mask)
save(img_otum,9)
img_otum2 = cv2.morphologyEx(img_otum,cv2.MORPH_OPEN,np.ones((5,5),dtype=np.uint8))
save(img_otum2,10)
if(rotate < np.pi):
    rotate_use = rotate+np.pi
if(rotate >= np.pi):
    rotate_use = rotate-np.pi

img_out2 = np.copy(img)
zhuanjiaos = np.zeros(8,np.float32)
for i in range(0,8):
    xu = dingxiang[i]['x']
    yu = dingxiang[i]['y']
    ru = min(dingxiang[:]['r'])
    img_zeros = np.zeros(img_otum2.shape,dtype=np.uint8)
    cv2.circle(img_zeros,(xu,yu),ru,255,-1)
    img_use_temp = cv2.bitwise_and(img_otum2,img_zeros)[yu-ru:yu+ru,xu-ru:xu+ru]
    save(img_use_temp,13+i)
    img_zeros = np.zeros(img_use_temp.shape,dtype=np.uint8)
    cv2.circle(img_zeros,(int(img_use_temp.shape[0]/2),int(img_use_temp.shape[1]/2)),int(2*ru/3),255,1)
    img_use_temp2 = cv2.bitwise_and(img_use_temp,img_zeros)
    yzz,xzz = np.where(img_use_temp2 > 100)
    xaz = np.average(xzz)
    yaz = np.average(yzz)
    (x,y,rotate_s) = getdeg([xaz-img_use_temp2.shape[0]/2,yaz-img_use_temp2.shape[0]/2,0])
    if rotate_s < rotate_use:
        rotate_s = rotate_s + 2*np.pi
        print i
    zhuanjiao = rotate_s - rotate_use
    print zhuanjiao
    zhuanjiaos[i] = 10*zhuanjiao/(2*np.pi)
    img_line3 = np.copy(cv2.cvtColor(img_use_temp,cv2.COLOR_GRAY2BGR))
    cv2.line(img_line3,(int(img_use_temp.shape[0]/2),int(img_use_temp.shape[1]/2)),(int(xaz),int(yaz)),(0,0,255),1)
    save(img_line3,21+i)
    
cv2.putText(img_out2,str(int(zhuanjiaos[0])),(int(dingxiang[0]['x']),int(dingxiang[0]['y'])),cv2.FONT_HERSHEY_SIMPLEX,1,np.random.randint(255,size=3),2)
for i in range(1,8):
    zs = int((zhuanjiaos[i]%1)*10)
    if zs < 2 and zhuanjiaos[i-1] > 5:
        zhuanjiaos[i] = zhuanjiaos[i] - 1
    if zs > 8 and zhuanjiaos[i-1] < 5:
        zhuanjiaos[i] = zhuanjiaos[i] + 1
    if zhuanjiaos[i] > 10:
        zhuanjiaos[i] = zhuanjiaos[i] - 10
    cv2.putText(img_out2,str(int(zhuanjiaos[i])),(int(dingxiang[i]['x']),int(dingxiang[i]['y'])),cv2.FONT_HERSHEY_SIMPLEX,1,np.random.randint(255,size=3),2)
    
save(img_out2,123)
    
#img_otu2 = cv2.morphologyEx(img_otu,cv2.MORPH_CLOSE,np.ones((6,6),dtype=np.uint8))
#save(img_otu2,9)

"""
cv2.morphologyEx(img,cv2.MORPH_OPEN,np.ones((5,5),dtype=np.uint8)
lines = cv2.HoughLinesP()
"""
#依次求出直線度數

#依據正方向確定指針讀數





    
    
