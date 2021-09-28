import cv2
import numpy as np

img = cv2.imread('pic/pic1.jpg')

cv2.imshow('Вход', img)

rows, cols = img.shape[:2]

input_pts = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1]])
output_pts = np.float32([[cols - 1, 0], [0, 0], [cols - 1, rows - 1]])

M = cv2.getAffineTransform(input_pts, output_pts)

dst = cv2.warpAffine(img, M, (cols, rows))

out = cv2.hconcat([img, dst])
cv2.imwrite('pic/transform.png', out)
cv2.imshow('Вывод', out)
cv2.waitKey(0)

"""
getAffineTransform берет 3 точки из входного и выходного значения и заносит их в матрицу 2x3
"""