import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('img/face.jpg')

h,w,c = img.shape

## 샘플링 기법을 sampling_ratio에 나눠떨어지는 화소값만 읽어왔을 때

raw_img_list = []
for sampling_ratio in [1, 2, 4, 8, 16] :
    print( f'({h,w}) => ({h // sampling_ratio + 1 , w // sampling_ratio + 1})' )
    downsample = np.zeros((h // sampling_ratio + 1, w // sampling_ratio + 1, c), np.uint8)

    for j in range(0, h, sampling_ratio) :
        for i in range(0, w, sampling_ratio) :
            y = j // sampling_ratio
            x = i // sampling_ratio
            downsample[y,x, :] = img[j,i, :]

    resized_img = cv2.resize(downsample, (w,h))  ## 시각 비교를 위해 리사이즈
    raw_img_list.append(resized_img)

naive_downsample_imgs = np.hstack(raw_img_list)

cv2.imshow('Raw downsample', naive_downsample_imgs)
cv2.waitKey(0)
cv2.destroyAllWindows()
# 스무딩 후 다운샘플링

smoothing_filter_1d = np.array([[0.05, 0.25, 0.40, 0.25, 0.05]])
smoothing_filter_2d = smoothing_filter_1d.T @ smoothing_filter_1d
smoothing_filter_2d = np.dstack([smoothing_filter_2d] *3 )

pad_img = np.zeros((h+4, w+4, 3), np.uint8)
pad_img[2:h+2, 2:w+2, :] = img[:,:,:]
blur_img = np.zeros((h,w,3), np.uint8)

for j in range(2, h+2) :
    for i in range(2, w+2) :

        blur_img[j-2,i-2,:] = np.sum(pad_img[j-2:j+3, i-2:i+3,:] * smoothing_filter_2d, keepdims = 2)
        


smooth_img_list = []
for sampling_ratio in [1, 2, 4, 8, 16] :
    print( f'({h,w}) => ({h // sampling_ratio + 1 , w // sampling_ratio + 1})' )
    downsample = np.zeros((h // sampling_ratio + 1, w // sampling_ratio + 1, c), np.uint8)

    for j in range(0, h, sampling_ratio) :
        for i in range(0, w, sampling_ratio) :
            y = j // sampling_ratio
            x = i // sampling_ratio
            downsample[y,x, :] = pad_img[j,i, :]

    
    resized_img = cv2.resize(downsample, (w,h))  ## 시각 비교를 위해 리사이즈
    smooth_img_list.append(resized_img)

smooth_downsample_imgs = np.hstack(smooth_img_list)

cv2.imshow('Smooth&downsample', smooth_downsample_imgs)
cv2.waitKey(0)
cv2.destroyAllWindows()


compare = np.hstack( [raw_img_list[-1], smooth_img_list[-1]])
cv2.imshow('Raw vs Smoothe Downsampling with r = 16', compare)
cv2.waitKey(0)
cv2.destroyAllWindows()