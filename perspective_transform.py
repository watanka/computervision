import cv2
import numpy as np
import matplotlib.pyplot as plt

# 해당 이미지에서 모니터 이미지 정보를 읽어오고 싶다. opencv 내장함수 없이 perspective transform을 구현해보자.

# read image
img = cv2.imread('img/get_monitor_perspective.jpg')

font = cv2.FONT_HERSHEY_SIMPLEX
# 모니터의 네 꼭짓점을 수기로 기록한다.
def click_event(event, x,y, flags, params) :
    if event == cv2.EVENT_LBUTTONDOWN :
        print(x, y)
        cv2.putText(img, f'{(x,y)}', (x,y), font, 1, (255, 0, 0), 2)
        cv2.circle(img, (x,y), 2, (0,0,255), 3)
        cv2.imshow('image', img)

cv2.imshow('image', img)
cv2.setMouseCallback('image', click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()


# 네 꼭짓점 (x,y)
top_left = (279, 203)
top_right = (868, 431)
bottom_right = (864, 802)
bottom_left = (274, 855)

pts = np.array([top_left, top_right, bottom_right, bottom_left], dtype = np.float32)
# 이 꼭짓점에 포함되는 이미지 잘라서 확인하기

crop_img = img[int(min([p[1] for p in pts])) : int(max([p[1] for p in pts])), int(min([p[0] for p in pts])) : int(max([p[0] for p in pts]))] 

cv2.imshow('crop image', crop_img)
cv2.waitKey(0)
cv2.destroyAllWindows()



# 정면 시선으로 변환했을 때의 예상 좌표. 가로와 세로는 최대 - 최소
height = int(max([p[1] for p in pts]) - min([p[1] for p in pts]))
width = int(max([p[0] for p in pts]) - min([p[0] for p in pts]))

dst_pts = np.array([[0,0], [width, 0], [width, height], [0, height]], dtype = np.float32)


## 모니터 영역을 정면에서 보는 시선(perspective)으로 변환한다.
M = cv2.getPerspectiveTransform(src = pts, dst = dst_pts)
transform_img = cv2.warpPerspective(img, M = M, dsize = (width, height))

cv2.imshow('perspective transform_img', transform_img)
cv2.waitKey(0)
cv2.destroyAllWindows()