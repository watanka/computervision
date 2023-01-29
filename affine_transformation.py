import cv2
import numpy as np
import math

img = cv2.imread('img/face.jpg') 

theta = 20
rads = math.radians(theta)
# rotation_matrix = np.array([
#                             [np.cos(theta), -np.sin(theta), 0],
#                             [np.sin(theta),  np.cos(theta), 0],
#                             [0            ,            0 ,  1]
#                             ])

rot_img = np.uint8(np.zeros((img.shape)))

height, width = img.shape[:2]
midx, midy = (width//2, height//2)

for i in range(img.shape[1]) :
    for j in range(img.shape[0]) :
        x = (i - midx) * math.cos(rads) + (j - midy) * math.sin(rads)
        y = -(i - midx) * math.sin(rads) + (j - midy) * math.cos(rads)

        x = round(x) + midx
        y = round(y) + midy

        if (x>=0 and y>=0 and x < width and y < height) :
            rot_img[j,i,:] = img[y,x,:]


cv2.imshow(f'rotation : {theta}', rot_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite(f'img/face_rot{theta}.jpg', rot_img)