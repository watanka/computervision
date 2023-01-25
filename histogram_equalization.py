import cv2
import matplotlib.pyplot as plt
import numpy as np

og_img = cv2.imread('face.jpg')
gray_img = cv2.cvtColor(og_img, cv2.COLOR_BGR2GRAY)

cv2.imshow('original', og_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
## histogram equalization
## color range가 좁은 이미지의 range를 평평하게 펴줌으로써 선명도를 증가시킨다.
h, w = gray_img.shape[:2]

hist = [0 for _ in range(256)]
for j in range(h) :
    for i in range(w) :
        hist[gray_img[j,i]] += 1

for l in range(255) :
    hist[l] /= h*w

cum_hist = np.cumsum(hist)

equalized_img = np.zeros((h,w))

for j in range(h) :
    for i in range(w) :
        equalized_img[j,i] = round(cum_hist[gray_img[j,i]] * 255)

plt.imshow(np.hstack([gray_img, equalized_img]), cmap = 'gray')
plt.show()

# print(equalized_hist)
bins = np.linspace(0, 255)
plt.hist(gray_img.flatten().tolist(), bins, alpha = 0.5, label = 'gray')
plt.hist(equalized_img.flatten().tolist(), bins, alpha = 0.5, label = 'equalized')
plt.legend(loc ='upper right')
plt.show()



