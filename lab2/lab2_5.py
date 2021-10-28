import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

image = cv.imread("pic/pic1.jpg")
cv.imshow("image_before", image)

h,w = image.shape[:2]

image_to_YUV = cv.cvtColor(image, cv.COLOR_BGR2YUV)

size_blur=(5,5)

core=[[0.1, 0.1, 0.1],
    [0.1, 0.1, 0.1],
    [0.1, 0.1, 0.1]]
core_filter = np.array([
    [0.1, 0.1, 0.1],
    [0.1, 0.1, 0.1],
    [0.1, 0.1, 0.1]
])

a = (len(core) - 1) // 2
b = (len(core[0]) - 1) // 2

sum0=0
sum1=0
sum2=0

new_image=np.zeros((h,w,3), dtype=float)


count_for = (h*w)//9

for pic_y in range(h):
    for pic_x in range(w):
        for core_y in range(len(core)):
            for core_x in range(len(core[core_y])):
                sum0 += round(core[core_y ][core_x] * image[pic_x-core_y ][pic_y-core_x][0])
                sum1 += round(core[core_y][core_x] * image[pic_x - core_y][pic_y - core_x][1])
                sum2 += round(core[core_y][core_x] * image[pic_x - core_y][pic_y - core_x][2])
        new_image[pic_x][pic_y][0] = sum0
        new_image[pic_x][pic_y][1] = sum1
        new_image[pic_x][pic_y][2] = sum2
        sum0 = 0
        sum1 = 0
        sum2 = 0


cv.imshow("image_after_hand", new_image)
cv.imshow("image_after_def", cv.filter2D(image,-1,core_filter))
cv.imshow("image_after_blur", cv.blur(image, size_blur, cv.BORDER_DEFAULT))

cv.waitKey(0)