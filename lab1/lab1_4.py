# -*- coding: utf-8 -*-
import cv2 as cv
import numpy as np

image = cv.imread("pic/pic1.jpg")
cv.imshow("pic",image)

h, w = image.shape[:2]
center = w//2, h//2

#rotation
M = np.float32([[1, 0, 100], [0, 1, 100]])
image_dst = cv.warpAffine(image, M, (int(w*1.5),int(h*1.5)))
cv.imshow('Изображение, сдвинутое вправо и вниз', image_dst)

#поворот
angle = np.radians(20)

M = np.float32([[np.cos(angle), -(np.sin(angle)), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
cv.imshow('Поворот', cv.warpPerspective(image, M, (int(w*1.5),int(h*1.5))))
cv.imwrite('/pic/rotation.png'
'Rotation with warpPerspective(20).jpg', cv.warpPerspective(image, M, (w, h)))

center = (int(w / 2), int(h / 2))
rotation_matrix = cv.getRotationMatrix2D(center, -45, 0.6)
image_rotated = cv.warpAffine(image, rotation_matrix, (int(w*1.5),int(h*1.5)))
cv.imshow('Поворот', image_rotated)

#масштабирование
M1 = np.float32([ [2, 0  , 0],
              [0,   2, 0],
              [0,   0,   1] ])
image_scale = cv.warpPerspective(image,M1,(int(w*1.5),int(h*1.5)))
cv.imshow('Масштаб', image_scale)

#сдвиг
M = np.float32([ [1, 0.5, 0], [0, 1 , 0], [0, 0 , 1] ])
sheared_img = cv.warpPerspective(image,M,(int(w*1.5),int(h*1.5)))  # отключить оси x и y 28. plt.axis('off') 29. # show the resulting image 30. plt.imshow(sheared_img)
cv.imshow('Сдвиг', sheared_img)

cv.waitKey(0)