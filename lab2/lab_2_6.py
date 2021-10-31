import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

image = cv.imread("pic/pic1.jpg")
cv.imshow("image_before", image)
y,u,v = cv.split(image)
print("y",len(y[0]))
h,w = image.shape[:2]

image_to_YUV = cv.cvtColor(image, cv.COLOR_BGR2YUV)

flag=True

list_sum0 = []
list_sum1 = []
list_sum2 = []
list_list_sum0=[]
list_list_sum1=[]
list_list_sum2=[]


def pp(list_h, list_w):
    # sum0 = 0
    # sum1 = 0
    # sum2 = 0
    sum0 = []
    sum1 = []
    sum2 = []
    global core
    global list_sum0,list_sum1,list_sum2
    for i in range(len(list_h)):
        for j in range(len(list_w)):
            for k in range(3):
                sum0.append(image[list_h[i]][list_w[j]][0])
                sum1.append(image[list_h[i]][list_w[j]][1] )
                sum2.append(image[list_h[i]][list_w[j]][2] )

    sum0.sort()
    sum1.sort()
    sum2.sort()


    list_sum0.append(sum0[len(sum0)//2])
    list_sum1.append(sum1[len(sum1)//2])
    list_sum2.append(sum2[len(sum2)//2])

    if len(list_sum0) == 320:

        list_list_sum0.append(list_sum0)
        list_sum0 = []

    if len(list_sum1) == 320:
        list_list_sum1.append(list_sum1)
        list_sum1 = []

    if len(list_sum2) == 320:
        list_list_sum2.append(list_sum2)
        list_sum2 = []



list_h = np.arange(h)
list_w =np.arange(w)

for i in range(h*w):
    if len(list_w)>0:
        pp(list_h[:3],list_w[:3])
        list_w= np.delete(list_w,[0])
    elif len(list_w)==0:
        list_w=np.arange(w)
        list_h=np.delete(list_h,[0])




black = np.zeros((319,320,3), dtype=float)

black[:,:,0]=list_list_sum0
black[:,:,1]=list_list_sum1
black[:,:,2]=list_list_sum2


cv.imshow("after hand", black)
cv.imwrite('pic/bb2.jpg',black)

median = cv.medianBlur(image,5)
cv.imshow("after median", median)

cv.waitKey(0)