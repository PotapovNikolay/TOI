import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

image = cv.imread("pic/pic1.jpg")
cv.imshow("image_before", image)
y,u,v = cv.split(image)
print("y",len(y[0]))
h,w = image.shape[:2]

image_to_YUV = cv.cvtColor(image, cv.COLOR_BGR2YUV)

def mm(core,K):
    for k in range(len(core)):
        for l in range(len(core[k])):
            core[k][l]=core[k][l]*K
    return core

def mm_filter(core,K):
    new_core=np.array((5,5), dtype=float)
    for i in range(5):
        for j in range(5):
            new_core[i][j]=core[i][j]*K
    return new_core

#размытие
# core_filter = np.array([
#     [0.1, 0.1, 0.1],
#     [0.1, 0.1, 0.1],
#     [0.1, 0.1, 0.1]
# ])
# core_filter = np.array([[0.04, 0.04, 0.04, 0.04, 0.04],
#         [0.04, 0.04, 0.04, 0.04, 0.04],
#         [0.04, 0.04, 0.04, 0.04, 0.04],
#         [0.04, 0.04, 0.04, 0.04, 0.04],
#         [0.04, 0.04, 0.04, 0.04, 0.04]])

#по гауссу
# core_filter = np.array([[0.0625, 0.125, 0.0625],
#         [0.125, 0.25, 0.125],
#         [0.0625, 0.125, 0.0625]])
# core_filter = np.array([[1, 4, 7, 4, 1],
#                         [4, 16, 26, 16, 4],
#                         [7, 26, 41, 26, 7],
#                         [4, 16, 26, 16, 4],
#                         [1, 4, 7, 4, 1]])
#
#
# core_filter=core_filter.tolist()
#
# core_filter= mm(core_filter, 1 / 273)
# #
# core_filter=np.array(core_filter)
# print(type(core_filter))

#Резкость
# core_filter =  np.array([[0, -1, 0],
#         [-1, 5, -1],
#         [0, -1, 0]])

# core_filter = np.array([[1, 1, 1],
#         [1,-7, 1],
#         [1, 1, 1]])

core_filter = np.array([[-1, -1, -1],
        [-1, 9, -1],
        [-1, -1, -1]])

#dsd
#core_filter=np.eye(9)


flag=True

list_sum0 = []
list_sum1 = []
list_sum2 = []
list_list_sum0=[]
list_list_sum1=[]
list_list_sum2=[]



#размытие
# core = [[0.1, 0.1, 0.1],
#         [0.1, 0.1, 0.1],
#         [0.1, 0.1, 0.1]]

# core = [[0.04, 0.04, 0.04, 0.04, 0.04],
#         [0.04, 0.04, 0.04, 0.04, 0.04],
#         [0.04, 0.04, 0.04, 0.04, 0.04],
#         [0.04, 0.04, 0.04, 0.04, 0.04],
#         [0.04, 0.04, 0.04, 0.04, 0.04]]

#по гауссу
# core = [[0.0625, 0.125, 0.0625],
#         [0.125, 0.25, 0.125],
#         [0.0625, 0.125, 0.0625]]
# core = [[1, 4, 7, 4, 1],
#         [4, 16, 26, 16, 4],
#         [7, 26, 41, 26, 7],
#         [4, 16, 26, 16, 4],
#         [1, 4, 7, 4, 1]]
# core=mm(core,(1/273))

#Резкость
# core = [[0, -1, 0],
#         [-1, 5, -1],
#         [0, -1, 0]]

# core = [[1, 1, 1],
#         [1,-7, 1],
#         [1, 1, 1]]

core = [[-1, -1, -1],
        [-1, 9, -1],
        [-1, -1, -1]]

#оператор Собеля
# core = [[0,1,0],
#                 [1,-4,1],
#                 [0,1,0]]

#dsdsa
#core = np.eye(9).tolist()

print(core)

def pp(list_h, list_w):
    sum0 = 0
    sum1 = 0
    sum2 = 0
    global core
    global list_sum0,list_sum1,list_sum2
    for i in range(len(list_h)):
        for j in range(len(list_w)):
            for k in range(len(core)):
                for l in range(len(core[k])):
                    if i==k and j==l:
                        sum0 += int(round(image[list_h[i]][list_w[j]][0] * core[k][l]))
                        sum1 += int(round(image[list_h[i]][list_w[j]][1] * core[k][l]))
                        sum2 += int(round(image[list_h[i]][list_w[j]][2] * core[k][l]))

    list_sum0.append(sum0)
    list_sum1.append(sum1)
    list_sum2.append(sum2)

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
        pp(list_h[:len(core)],list_w[:len(core)])
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

cv.imshow("image_after_def", cv.filter2D(image,-1,core_filter))
#cv.imshow("image_after_blur", cv.blur(image, (5,5), cv.BORDER_DEFAULT))
#cv.imshow("image_after_gaussian", cv.GaussianBlur(image,(5,5),cv.BORDER_DEFAULT))

cv.waitKey(0)