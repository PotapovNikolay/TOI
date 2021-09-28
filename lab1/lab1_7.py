import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('pic/pic2.png')

cv2.imshow('Вход', img)

img_copy = np.copy(img)

img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)

input_pts = np.float32([[32, 0], [32, 320], [285, 285], [289, 35]])
output_pts = np.float32([[0, 0], [0, 320], [320, 320], [320, 0]])

M = cv2.getPerspectiveTransform(input_pts, output_pts)

out = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)

cv2.imwrite('pic/Perspective.png', out)

cv2.imshow('Вывод', out)
cv2.waitKey(0)
