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

plt.subplot(121), plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB)), plt.title('исходное изображение'), plt.axis('off')



image=cv.cvtColor(image, cv.COLOR_BGR2RGB)

image =  cv.GaussianBlur(image,(5,5),cv.BORDER_DEFAULT)

image=cv.cvtColor(image, cv.COLOR_RGB2GRAY)


dst = cv.Laplacian(image, cv.CV_16S, ksize=5)
Laplacian = cv.convertScaleAbs(dst)

plt.subplot (122), plt.imshow (Laplacian, cmap = plt.cm.gray), plt.title ('оператор Лапласа'), plt.axis ('off')

plt.show()

print(image[0][0])

# core_laplacian = [
#                 [0,1,0],
#                 [1,-4,1],
#                 [0,1,0]]
core_laplacian = [
                [0,0,-1,0,0],
                [0,-1,-2,-1,0],
                [-1,-2,-17,-2,-1],
                [0,-1,-2,-1,0],
                [0,0,-1,0,0]]



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
    sum=[]
    p_sum=[]

    global core
    global list_sum0,list_sum1,list_sum2
    for i in range(len(list_h)):
        for j in range(len(list_w)):
            for k in range(len(core_laplacian)):
                for l in range(len(core_laplacian[k])):
                    if i==k and j==l:
                        sum0+=image[list_h[i]][list_w[j]] * core_laplacian[k][l]

    list_sum0.append(sum0)

    if len(list_sum0) == 320:

        list_list_sum0.append(list_sum0)
        list_sum0 = []


list_h = np.arange(h)
list_w =np.arange(w)

for i in range(h*w):
    if len(list_w)>0:
        pp(list_h[:len(core_laplacian)],list_w[:len(core_laplacian)])
        list_w= np.delete(list_w,[0])
    elif len(list_w)==0:
        list_w=np.arange(w)
        list_h=np.delete(list_h,[0])



print(len(list_list_sum0))


black = np.zeros((319,320,1), dtype=float)



black[:,:,0]=list_list_sum0



black=black.astype('float32')

# for i in range(319):
#     for j in range(320):
#         image[i][j] +=-1*black[i][j]
#         black[i][j]=image[i][j]

cv.imshow("after hand", cv.cvtColor(black, cv.COLOR_GRAY2BGR))
cv.imwrite('pic/bb3.png',cv.cvtColor(black, cv.COLOR_GRAY2BGR))




cv.waitKey(0)