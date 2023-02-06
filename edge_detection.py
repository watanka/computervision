import cv2
import numpy as np

img = cv2.imread('img/go_table.jpg', cv2.IMREAD_GRAYSCALE)
h,w = img.shape[:2]



dx = cv2.Sobel(img, -1, 1, 0, delta=128)
dx = cv2.convertScaleAbs(dx)

dy = cv2.Sobel(img, -1, 0, 1, delta=128)
dy = cv2.convertScaleAbs(dy)

grad = cv2.addWeighted(dx, 0.5, dy, 0.5, 0)

cv2.imshow('dx', dx)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow('dy', dy)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow('dx + dy', grad)
cv2.waitKey(0)
cv2.destroyAllWindows()
