import numpy as np
import scipy as scp
import cv2
import sys
import matplotlib.pyplot as plt
global refPt
refPt=[]

def euclidean(x,y):
    d=0
    for i in range(3):
        d+=np.power(x[i]-y[i],2.0)
    return np.sqrt(d)

def manhattan(x,y):
    d=0
    for i in range(3):
        d+=np.abs(x[i]-y[i])
    return d
def chebyshev(x,y):
    d=0
    for i in range(3):
        if d<np.abs(x[i]-y[i]):
            d=np.abs(x[i]-y[i])
    return d
def spearman(x,y):
    d=0
    for i in range(3):
        d+=np.power(x[i]-y[i],2.0)
    return d

def Kmeans(m,c,n,max=1,chose=0):
    x,y,z=m.shape
    K=np.ones((x,y))
    J=np.array([1000,23232323])
    count=0
    distancias=np.zeros(n)
    num=np.zeros((n,z))
    dem=np.zeros(n)
    while(np.abs(J[0]-J[1])>3000 and count<max):
        J[0]=J[1]
        J[1]=0
        for i in range(x):
            for j in range(y):
                for k in range(n):
                    if chose==0:
                        distancias[k]=euclidean(m[i][j],c[k])
                    elif chose==1:
                        distancias[k]=manhattan(m[i][j],c[k])
                    elif chose==2:
                        distancias[k]=chebyshev(m[i][j],c[k])
                    elif chose==3:
                        distancias[k]=spearman(m[i][j],c[k])
                pos=scp.argmin(distancias)
                K[i][j]=pos
                J[1]+=distancias[pos]
                num[pos]+=distancias[pos]*m[i][j]
                dem[pos]+=distancias[pos]*1.0
        for i in range(0,n):
            if dem[i]==0:
                c[i]=0
            else:
                c[i]=num[i]/dem[i]
            num[i]=0
            dem[i]=0
        count+=1
    return K
def click_and_crop(event, x, y, flags, param):
    global refPt
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt.append([x, y])

def Clickcenters(image):
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", click_and_crop)
    while True:
        cv2.imshow("image", image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("c"):
            break

def click_and_crop(event, x, y, flags, param):
    global refPt
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt.append([x, y])

def Center(img):
    l=len(refPt)
    c=np.zeros((l,3))
    for i in range(l):
        for j in range(3):
            c[i][j]=img[refPt[i][1]][refPt[i][0]][j]
    return c

img=cv2.imread('image_color/3096.jpg')
Clickcenters(img)
cv2.destroyAllWindows()
c=Center(img)
K=Kmeans(img,c,len(refPt))
plt.imshow(K)
plt.show()
