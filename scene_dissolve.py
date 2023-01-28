import cv2
import numpy as np

'''첫번째 이미지와 두번째 이미지를 alpha로 블렌딩함. scene dissolve 구현'''
img1 = cv2.imread('img/face.jpg')
img2 = cv2.imread('img/face2.jpg')

img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

for i in range(0, 10) :
    alpha = i/10
    # frame = cv2.addWeighted(img2, alpha, img1, 1-alpha, 0)
    # cv2.imshow(f'alpha = {alpha}', frame)
    cv2.imshow(f'alpha = {alpha}', np.uint8(img1 * (1-alpha) + img2 * alpha)) ## cv2 버젼이 2x 빠름
    cv2.waitKey(0)
    cv2.destroyAllWindows()