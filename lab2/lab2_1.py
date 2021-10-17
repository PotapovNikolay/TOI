import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

image = cv.imread('pic/ap.jpg')
#cv.imshow("pic",image)
h,w = image.shape[:2]

RGB_image = cv.cvtColor(image,cv.COLOR_BGR2RGB)
blue, green, red = cv.split(image)

#cv.imshow("sd",blue)
colors = ['#FF7382','#9CFF8C','#75B1FF']

plt.hist([red.ravel(),green.ravel(),blue.ravel()], 256, [0, 256], color=[colors[0], colors[1], colors[2]])
plt.title ("Гистограмма plt")

hist_blue = cv.calcHist([blue], [0], None, [256], [0, 256])
hist_green = cv.calcHist([green], [0], None, [256], [0, 256])
hist_red = cv.calcHist([red], [0], None, [256], [0, 256])
#hist_cv=cv.calcHist([image],[0,1,2], None, [256,256,256], [0, 256,0, 256,0, 256] )

plt.figure ()
plt.title ("Гистограмма opencv")
plt.plot (hist_blue,colors[2])
plt.plot(hist_green,colors[1])
plt.plot(hist_red,colors[0])

axis_x_r = np.zeros((256), dtype=int)
axis_x_g = np.zeros((256), dtype=int)
axis_x_b = np.zeros((256), dtype=int)

p= np.arange(256)

for i in range(h):
    for j in range(w):
        axis_x_r[image[i][j][0]]+=1
        axis_x_g[image[i][j][1]] += 1
        axis_x_b[image[i][j][2]] += 1

for i in range(256):
    axis_x_r[i]/=100
    axis_x_g[i]/=100
    axis_x_b[i] /= 100
r = np.c_[p,axis_x_r]
g=np.c_[p,axis_x_g]
b=np.c_[p,axis_x_b]

M1 = np.float32([ [2, 0  , 0],
              [0,   1, 0],
              [0,   0,   1] ])
black = np.ones((1400,256,3), dtype=float)
hist_hand_r = cv.warpPerspective(cv.polylines(black,[r],False,(255, 0, 0),1),M1,(256*2,int(1400/3)))
hist_hand_g = cv.warpPerspective(cv.polylines(black,[g],False,(0, 255, 0),1),M1,(256*2,int(1400/3)))
hist_hand_b = cv.warpPerspective(cv.polylines(black,[b],False,(0, 0, 255),1),M1,(256*2,int(1400/3)))
cv.imshow("hand hist",hist_hand_r[::-1,:,:])
cv.imshow("hand hist",hist_hand_g[::-1,:,:])
cv.imshow("hand hist",hist_hand_b[::-1,:,:])


plt.show()
cv.waitKey(0)