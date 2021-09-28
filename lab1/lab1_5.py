import numpy as np
import cv2 as cv
import math

def matrixmult(m1,m2):
    s=0
    t=[]
    m3=[]
    if len(m2)!=len(m1[0]):
        print ("Матрицы не могут быть перемножены")
    else:
        r1=len(m1)
        c1=len(m1[0])
        c2=len(m2[0])
        for z in range(0,r1):
            for j in range(0,c2):
                for i in range(0,c1):
                   s=s+m1[z][i]*m2[i][j]
                t.append(s)
                s=0
            m3.append(t)
            t=[]
    return m3

img = cv.imread('pic/pic1.jpg')
h, w = img.shape[:2]
cv.imshow('Ввод', img)
black = np.ones((700,700,3), dtype=float)
ll = np.ones((1,3), dtype=int)
C=[]


#передвижение

T=np.array([
    [1,0,0],
    [0, 1,0],
    [50, 50,1]
])

#масштабирование

# T=np.array([
#     [2,0,0],
#     [0, 2,0],
#     [0, 0,1]
# ])

#поворот

# T=np.array([
#     [np.cos(np.pi/6),np.sin(np.pi/6),0],
#     [-np.sin(np.pi/6), np.cos(np.pi/6),0],
#     [0, 0,1]
# ])


#сдвиг

# T=np.array([
#     [1,np.tan(np.pi/4),0],
#     [np.tan(np.pi/6), 1,0],
#     [0, 0,1]
# ])


for i in range(1,h+1):
    for j in range(1,w+1):
        ll[0,0] = j
        ll[0,1] = i
        C += matrixmult(ll, T)
        black[round(C[0][1])][round(C[0][0])][0] = img[i - 1][j - 1][0]
        black[round(C[0][1])][round(C[0][0])][1] = img[i - 1][j - 1][1]
        black[round(C[0][1])][round(C[0][0])][2] = img[i - 1][j - 1][2]
        C.clear()


#передвижение и масштаб
# for row in NY:
#     for col in NX:
#         for color in range(3):
#             black[row][col][color] = img[NY.index(row)][NX.index(col)][color]


#поворот
# for i in range(1,h+1):
#     for j in range(1,w+1):
#         print(NY)
#         black[NY[0]][NX[0]][0] = img[i - 1][j - 1][0]
#         black[NY[0]][NX[0]][1] = img[i - 1][j - 1][1]
#         black[NY[0]][NX[0]][2] = img[i - 1][j - 1][2]




#сдвиг
# for row in NY:
#     for col in NX:
#         for color in range(3):
#             if len(NY)>len(NX):
#                 black[row][col][color] = img[round(NY.index(row) // 3)][NX.index(col)][color]
#             else:
#                 black[row][col][color] = img[NY.index(row)][round(NX.index(col)//3)][color]

cv.imwrite('pic/tt5.png', black)
cv.waitKey(0)