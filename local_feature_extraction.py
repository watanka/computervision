import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('img/local_feature.jpg', cv2.IMREAD_GRAYSCALE)


# 모니터의 네 꼭짓점을 수기로 기록한다.
font = cv2.FONT_HERSHEY_SIMPLEX

def click_event(event, x,y, flags, params) :
    if event == cv2.EVENT_LBUTTONDOWN :
        
        print(x, y)
        cv2.putText(img, f'{(x,y)}', (x,y), font, 1, (255, 0, 0), 2)
        cv2.circle(img, (x,y), 2, (0,0,255), 3)
        cv2.imshow('image', img)

# Moravec's corner detection
# 수기로 SSD(Sum of Squared Difference)를 확인하고 싶은 좌표 추출.

cv2.imshow('image', img)
cv2.setMouseCallback('image', click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()

a, b, c = (332, 520), (453, 243), (189, 199)

def calc_SSD(img, pt) :
    '''
    calculate SSD(Sum of Squared Difference) from Moravec's Corner detection.
    값이 클수록, 마스크 내에 그래디언트값이 크다는 뜻이고, 엣지일 확률이 높다는 뜻이다.
    '''
    res = np.zeros((3,3), np.uint8)
    y,x = pt
    for j in range(-1, 2) :
        for i in range(-1, 2) :
            v, u = y + j, x + i
            res[j+1,i+1] = np.sum(np.square(img[y-1:y+2,x-1:x+2] - img[v-1:v+2, u-1:u+2]))
    return np.sum(res)

print('SSD(a) : ', calc_SSD(img, a))
print('SSD(b) : ', calc_SSD(img, b))
print('SSD(c) : ', calc_SSD(img, c))
