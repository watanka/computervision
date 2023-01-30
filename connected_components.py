import cv2
import matplotlib.pyplot as plt
import numpy as np


img = cv2.imread('img/figures.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, binary_img = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)
binary_img = np.array(binary_img, np.int8) # cv2 기본값은 uint8(unsigned int 2**8 = 256)으로 0-255의 양수 값만 갖는다. -1을 넣어도 값이 변하지 않는다.

h,w = binary_img.shape[:2]



binary_img[binary_img == 255] = -1

# brute execution : 이미지 화소가 많아지면 stack overflow 발생
def floodfill(img, j, i, label) :    
    if 0<j<h and 0<i<w :
        if img[j,i] == -1 :
            img[j,i] = label
            floodfill(img, j-1, i, label)
            floodfill(img, j+1, i, label)
            floodfill(img, j, i-1, label)
            floodfill(img, j, i+1, label)


# 메모리 적게 사용하는 버젼
def efficient_floodfill(img, j, i, label) :
    queue = []
    queue.append((j,i))

    while queue :
        y,x = queue.pop(0)
        if img[y,x] == -1 :
            left = x
            right = x
            while img[y, left - 1] == label :
                left -= 1
            while img[y, right + 1] == label :
                right += 1

            for c in range(left, right+1) :
                img[y,c] = label
                if img[y-1, c] == -1 and (c==left or img[y-1, c-1] != -1) :
                    queue.append((y-1, c))
                if img[y+1, c] == -1 and (c==left or img[y+1, c-1] != -1) :
                    queue.append((y+1, c))


label = 1
for j in range(1, h-2) :
    for i in range(1, w-2) :
        if binary_img[j,i] == -1 :
            efficient_floodfill(binary_img, j,i,label)
            label += 1

print(np.unique(binary_img))

# cv2.imshow('flood fill img', binary_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

