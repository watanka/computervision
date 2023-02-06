import cv2
import numpy as np

img = cv2.imread('img/go_table.jpg', cv2.IMREAD_GRAYSCALE)
h,w = img.shape[:2]


sobel_edge_operator_x = np.array([[-1, 0, 1],
                                  [-2, 0, 2],
                                  [-1, 0, 1]])

sobel_edge_operator_y = np.array([[-1, -2, -1],
                                  [ 0,  0,  0],
                                  [ 1,  2,  1]])


# 3x3 연산을 위해 이미지 테두리에 0 패딩을 추가한다.
pad_img = np.zeros((h+2, w+2), np.uint8)
pad_img[1:-1, 1:-1] = img

cv2.imshow('pad_img', pad_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


dx = np.zeros((h+2, w+2), np.uint8)
dy = np.zeros((h+2, w+2), np.uint8)


for j in range(1, h+1) :
    for i in range(1, w+1) :

        dx[j,i] = np.sum(pad_img[j-1:j+2, i-1:i+2] * sobel_edge_operator_x)
        dy[j,i] = np.sum(pad_img[j-1:j+2, i-1:i+2] * sobel_edge_operator_y)

## TODO : 값을 직접 계산하면, 노이즈가 심하다. 반면, cv2 모듈을 사용하면, 비교적 깔끔한 값이 나온다. 어떤 다른 점이 있는걸까?


dx = cv2.Sobel(img, -1, 1, 0, delta=128)
dy = cv2.Sobel(img, -1, 0, 1, delta=128)

combined = dx + dy

cv2.imshow('dx', dx)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow('dy', dy)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow('combined', combined)
cv2.waitKey(0)
cv2.destroyAllWindows()
