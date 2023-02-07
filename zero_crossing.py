import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

img = cv2.imread('img/crossfit.jpg', cv2.IMREAD_GRAYSCALE)

h,w = img.shape

cv2.imshow('original image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
## gaussian filter w/ sigma

def get_gaussian_filter(kernel_size, sigma) :
    center = kernel_size // 2
    x, y = np.mgrid[-center : kernel_size - center, -center : kernel_size - center]
    
    g = 1 / (2*np.pi * sigma) * np.exp(-(np.square(x) + np.square(y)) / (2 * (sigma ** 2)))

    return g

fig, ax = plt.subplots(nrows =2, ncols = 2, subplot_kw = {'projection' : '3d'})

kernel_size = 30

# sigma = 0.5
# x, y = np.mgrid[:kernel_size, :kernel_size]
# z = get_gaussian_filter(kernel_size, sigma)


# ax[0,0].plot_surface(x,y,z, cmap=cm.coolwarm) 

# sigma = 3
# x, y = np.mgrid[:kernel_size, :kernel_size]
# z = get_gaussian_filter(kernel_size, sigma)
# ax[0,1].plot_surface(x,y,z, cmap=cm.coolwarm) 

# sigma = 5
# x, y = np.mgrid[:kernel_size, :kernel_size]
# z = get_gaussian_filter(kernel_size, sigma)
# ax[1,0].plot_surface(x,y,z, cmap=cm.coolwarm) 

# sigma = 8
# x, y = np.mgrid[:kernel_size, :kernel_size]
# z = get_gaussian_filter(kernel_size, sigma)
# ax[1,1].plot_surface(x,y,z, cmap=cm.coolwarm) 

# fig.tight_layout()
# plt.show()

kernel_size = 10
sigma = 0.5
gaussian_filter = get_gaussian_filter(kernel_size, sigma)
## Laplacian filter
## x와 y의 2차 편미분값의 합
laplacian_filter = np.array([[0,  1,  0],
                             [1, -4,  1],
                             [0,  1,  0]])

## LOG (Laplacian of Gaussian) filter
# gaussian_filter_img = cv2.filter2D(img, -1, kernel = gaussian_filter) # 앞으로 y,x 좌표를 for loop로 도는 대신 filter2D 함수를 사용한다.
# cv2.imshow('gaussian filterd image', gaussian_filter_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# log_filter_img = cv2.filter2D(gaussian_filter_img, -1, kernel = laplacian_filter)
# cv2.imshow('LoG filterd image', log_filter_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

## 연산 부담을 줄이기 위해서 Laplacian과 Gaussian 필터를 먼저 결합한 후 적용도 가능하다.
center = kernel_size // 2
y,x = np.mgrid[-center : kernel_size - center, -center : kernel_size - center]
log_filter = ((y**2 + x**2 - (2.0*sigma**2)) / sigma ** 4) * np.exp(-(y**2 + x**2) / (2.0*sigma**2)) / (1 / (2.0 * np.pi * sigma ** 2))

log_img = cv2.filter2D(img, -1, kernel = log_filter)
cv2.imshow('LoG filterd image', log_img)
cv2.waitKey(0)
cv2.destroyAllWindows()



## find out zero crossing

zero_crossing = np.zeros(img.shape, np.uint8)

for j in range(1, h-1) :
    for i in range(1, w-1) :
        if log_img[j][i] == 0 :
            if (log_img[j][i-1] < 0 and log_img[j][i+1] > 0) or (log_img[j][i-1] > 0 and log_img[j][i+1] < 0) or \
                (log_img[j-1][i] < 0 and log_img[j+1][i] > 0) or (log_img[j-1][i] > 0 and log_img[j+1][i] < 0) or \
                    (log_img[j-1][i-1] < 0 and log_img[j+1][i+1] > 0) or (log_img[j-1][i-1] > 0 and log_img[j+1][i+1] < 0) or \
                        (log_img[j-1][i+1] < 0 and log_img[j+1][i-1] > 0) or (log_img[j-1][i+1] > 0 and log_img[j+1][i-1] < 0) : ## 동서, 북서 
                        zero_crossing[j][i] = 255.
        if log_img[j][i] < 0 :
            if (log_img[j][i-1] > 0) or (log_img[j][i+1] > 0) or (log_img[j-1][i] > 0) or (log_img[j+1][i] > 0) :
                zero_crossing[j][i] = 255.


cv2.imshow('zero crossing', zero_crossing)
cv2.waitKey(0)
cv2.destroyAllWindows()