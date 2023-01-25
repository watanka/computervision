import cv2
import numpy as np
import matplotlib.pyplot as plt


model_img = cv2.imread('face.jpg')
model_hsv_img = cv2.cvtColor(model_img, cv2.COLOR_BGR2HSV)

curr_img = cv2.imread('face2.jpg')
curr_hsv_img = cv2.cvtColor(curr_img, cv2.COLOR_BGR2HSV)

cv2.imshow('original', model_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

q = 64
L = 256

def quantize(a, q = q, L = L) :
    '''
    a : pixel value
    q : scale
    L : original range for Hue or Saturation
    '''
    return int(np.floor(a*q / L))

def get_hist(img, q) :
    # hue, saturation histogram 생성
    height, width = img.shape[:2]
    hist = [[0 for _ in range(q)] for _ in range(q)] 
    for j in range(height) :
        for i in range(width) :
            hist[quantize(img[j,i][0])][quantize(img[j,i][1])] += 1  # hue, saturation

    for j in range(q) :
        for i in range(q) :
            hist[j][i] /= (height * width) # 정규화
    return hist

model_hist = get_hist(model_hsv_img, q)
curr_hist = get_hist(curr_hsv_img, q)

def hist_rate(model_hist, curr_hist, hueval, satval) :
    return min( model_hist[hueval][satval] / curr_hist[hueval][satval], 1.0)

# back projection
height, width = curr_hsv_img.shape[:2]
o = np.zeros((height,width)) # 가능성 맵



for j in range(height) :
    for i in range(width) :
        pixel = curr_hsv_img[j,i]
        o[j,i] = hist_rate(model_hist, curr_hist, hueval = quantize(pixel[0]), \
                                                  satval = quantize(pixel[1]))

plt.hist(o.flatten().tolist())
plt.imshow(o * 255, cmap = 'gray')
plt.show()


