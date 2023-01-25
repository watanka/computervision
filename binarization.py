import cv2
import numpy as np
import time

# naive
img =  cv2.imread('img/face.jpg')
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

height, width = gray_img.shape[:2]

variations = []

naive_otsu_start = time.time()
for T in range(256) :
    white = gray_img[gray_img >= T]
    black = gray_img[gray_img < T]

    # print(len(white))
    # print(len(black))
    white_var = np.var(white) 
    if len(white) == 0 :
        white_var = 0
    black_var = np.var(black)
    if len(black) == 0 :
        black_var = 0
    v_within = len(white) * white_var + len(black) * black_var
    variations.append(v_within)

best_threshold = np.argmin(variations)          
naive_otsu_elapse_time = time.time() - naive_otsu_start

print(f'best threshold that has minimum variance is {variations[best_threshold]} when T={best_threshold}' )

_, naive_threshed = cv2.threshold(gray_img, best_threshold, 255, cv2.THRESH_BINARY)
_, otsu_threshed = cv2.threshold(gray_img, -1, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)


# OTSU algorithm : more efficient version ## TODO : 값이 naive와 다름. 뭔가 잘못됨.
## 순환식으로 T-1에 대해 구한 값을 활용하여 연산량을 줄임.
## w0(t) = w0(t-1) + h(t)
## mu_0(t) = (w0(t-1)*mu_0(t-1) + t*h(t)) / w0(t)
## mu_1(t) = (mu - w0(t) * mu_0(t)) / (1-w0(t))
## v_between = w_0(t)(1-w_0(t))(mu_0(t) - mu_1(t))**2. v_between이 최대값인 T를 찾아라.

otsu_time = time.time()
mu_total = np.mean(gray_img)
v_total = np.var(gray_img)

variations = []

height, width = gray_img.shape[:2]
prev_w0 = len(gray_img[gray_img == 0]) / (height * width)
prev_mu0 = 0
prev_mu1 = (mu_total - prev_w0 * prev_mu0) / (1- prev_w0)
v_between = prev_w0 * ( 1- prev_w0) * ((prev_mu0 - prev_mu1)**2)
variations.append(v_between)

for t in range(1, 256) :

    h_t = len(gray_img[gray_img == t] / (height * width))
    curr_w0 = prev_w0 + h_t
    curr_mu0 = (prev_w0 *  prev_mu0 + t * h_t) / curr_w0
    curr_mu1 = (mu_total - curr_w0 * curr_mu0) / (1 - curr_w0)

    v_between = curr_w0 * (1 - curr_w0) * ((curr_mu0 - curr_mu1)**2)
    variations.append(v_between)

    prev_w0 = curr_w0
    prev_mu0 = curr_mu0

best_threshold = np.argmax(variations)

otsu_elapse_time = time.time() - otsu_time

print(f'best threshold that has maximum v_between is {variations[best_threshold]} when T={best_threshold}' )


print(f'naive time : {naive_otsu_elapse_time}\notsu algorithm time : {otsu_elapse_time}')

cv2.imshow('naive search otsu vs cv2 otsu', np.hstack([naive_threshed, otsu_threshed]))
cv2.waitKey(0)
cv2.destroyAllWindows()