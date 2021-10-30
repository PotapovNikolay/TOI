import math

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

image = cv.imread("pic/pic1.jpg")
cv.imshow("image_before", image)
y,u,v = cv.split(image)
print("y",len(y[0]))
h,w = image.shape[:2]
print(image[0][0])

image_sobol = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
sobelx = cv.Sobel(image_sobol,cv.CV_64F,1,0,ksize=5)
sobely = cv.Sobel(image_sobol,cv.CV_64F,0,1,ksize=5)
sobel = cv.Sobel(image_sobol,cv.CV_64F,1,1,ksize=5)

plt.subplot(2,2,1),plt.imshow(image_sobol,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(sobel,cmap = 'gray')
plt.title('Sobel '), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])


plt.show()

image=cv.cvtColor(image, cv.COLOR_BGR2GRAY)


core_sobol_y=[[-1,0,1],
              [-2,0,2],
              [-1,0,1]]

core_sobol_x=[[-1,-2,-1],
              [0,0,0],
              [1,2,1]]

flag=True

list_sum0 = []
list_sum1 = []
list_sum2 = []
list_list_sum0=[]
list_list_sum1=[]
list_list_sum2=[]

def pp(list_h, list_w):
    sum0 = 0
    sum1 = 0
    sum2 = 0
    sum0_x=0
    sum_low=0
    sum_hight=0
    sum1_x=0
    sum2_x=0
    sum0_y=0
    sum1_y = 0
    sum2_y = 0
    global core
    global list_sum0,list_sum1,list_sum2
    for i in range(len(list_h)):
        for j in range(len(list_w)):
            for k in range(len(core_sobol_x)):
                for l in range(len(core_sobol_x[k])):
                    if i==k and j==l:
                        sum0_x += round(image[list_h[i]][list_w[j]] * core_sobol_x[k][l])
                        sum0_y += round(image[list_h[i]][list_w[j]] * core_sobol_y[k][l])

    sum0 = math.sqrt((sum0_x ** 2) + (sum0_y ** 2))

    if sum0>255:
        sum0=255
    list_sum0.append(sum0)

    if len(list_sum0) == 320:

        list_list_sum0.append(list_sum0)
        list_sum0 = []


list_h = np.arange(h)
list_w =np.arange(w)

for i in range(h*w):
    if len(list_w)>0:
        pp(list_h[:len(core_sobol_x)],list_w[:len(core_sobol_x)])
        list_w= np.delete(list_w,[0])
    elif len(list_w)==0:
        list_w=np.arange(w)
        list_h=np.delete(list_h,[0])



print(len(list_list_sum0))
print(list_list_sum0)

black = np.zeros((319,320,1), dtype=float)

black[:,:,0]=list_list_sum0


black=black.astype('float32')

cv.imshow("after hand", cv.cvtColor(black, cv.COLOR_GRAY2BGR))
cv.imwrite('pic/bb3.png',cv.cvtColor(black, cv.COLOR_GRAY2BGR))




cv.waitKey(0)