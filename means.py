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

def Cmeans(X,C,n,max=1,chose=0): #X: Foto input, C: centroides, n total de cluster, max: maximo de iteraciones
    x,y,z=X.shape
    K=np.ones((x,y)) #segmentate foto
    J=np.zeros(2) #FUncion de # OPTI
    U=np.zeros((x,y,n)) #Membership Matrix
    count=0 #contador de iteraciones
    distancias=np.zeros((x,y,n)) #distancia entre cada punto con respecto al centroides
    C_dem=np.zeros((n)) #Calculo de centroides
    power=(1.0/(1.0-n))
    C_num=np.zeros((n,3)) #Calculo de centroides
    while(count<max):
        J[0]=J[1]
        for i in range(x):
            for j in range(y):
                u_dem=0 #demonimador membership
                for k in range(n):
                    if chose==0:
                        distancias[i][j][k]=euclidean(X[i][j],C[k])
                    elif chose==1:
                        distancias[i][j][k]=manhattan(X[i][j],C[k])
                    elif chose==2:
                        distancias[i][j][k]=chebyshev(X[i][j],C[k])
                    elif chose==3:
                        distancias[i][j][k]=spearman(X[i][j],C[k])
                    if distancias[i][j][k]>0:
                        U[i][j][k]=np.power(distancias[i][j][k],power)
                        u_dem+=U[i][j][k] #sumatori en k((distancias Xij,Ck)^coeficient)
                for k in range(n):
                    U[i][j][k]=U[i][j][k]/u_dem #memebership value
                    C_num[k]+=np.power(U[i][j][k],n)*X[i][j] #sum Uijk^n*Xij
                    C_dem[k]+=np.power(U[i][j][k],n)
                    J[1]+=np.power(U[i][j][k],n)*distancias[i][j][k]
                pos=scp.argmax(U[i][j])
                K[i][j]=pos
        #print "Membership Matrix:",U
        #print "Optimal error: ",np.abs(J[0]-J[1])
        #print "Prueba numero: ",count
        for k in range(0,n):
            C[k]=C_num[k]/C_dem[k]
            #print "denominadores: ",C_num[k],C_dem[k]
            #C_num[k]=0
            #C_dem[k]=0
        #print "Nuevos Centroides:",C
        count+=1
    return K

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
        #print c
        #print J
        #print count
    return K
def readfile(P):
    if len(P)==1:
        src='image_color/3096.jpg'
        src1='ground/3096.png'
    else:
        src='image_color/'+P[1]+'.jpg'
        src1='ground/'+P[1]+'.png'
    return P[1],cv2.imread(src),cv2.imread(src1,0)

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

def Center(img):
    l=len(refPt)
    c=np.zeros((l,3))
    for i in range(l):
        for j in range(3):
            c[i][j]=img[refPt[i][1]][refPt[i][0]][j]
        c[i]=prom(img,refPt[i][1],refPt[i][0])
    return c

def Miserror(imgData,imgCK):
    rows,cols = imgCK.shape
    A=np.zeros((rows,cols))
    for i in range(rows):
        for j in range(cols):
            if imgData[i][j]>125:
                A[i][j]=1
    diff = np.abs(A - imgCK)
    k=0
    for i in range(rows):
        for j in range(cols):
            k+=diff[i][j]
    print 'Error: ',(k*1.00)/(rows*cols)

def prom(M):
    x,y,z=M.shape
    K=np.zeros((x,y,z))
    for i in range(x-1):
        for j in range(y-1):
            for p in range(i-1,i+1):
                for q in range(j-1,j+1):
                    K[i][j]+= M[p][q]
            K[i][j]=K[i][j]/9
    print K
    return K

src,img,img1=readfile(sys.argv)
img=prom(img)
Clickcenters(img)
cv2.destroyAllWindows()
c=Center(img)
print 'Centers:',c
test=[1,3,5]
for n in test:
    print 'Prueba:',n
    print 'Cmeans'
    Cdir='C'+src
    for i in range(0,4):
        C=Cmeans(img,c,len(c),n,i)
        dir='results/'+src+'/Cmeans/'+Cdir+'_'+str(n)+'_'+str(i)+'.png'
        Miserror(img1,C)
    print 'Kmeans'
    Kdir='K'+src
    for i in range(0,4):
        K=Kmeans(img,c,2,n,i)
        dir='results/'+src+'/Kmeans/'+Kdir+'_'+str(n)+'_'+str(i)+'.png'
        cv2.imwrite(dir,K*255)
        Miserror(img1,K)
