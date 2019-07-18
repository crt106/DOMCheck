import numpy as np
import cv2

jpgname1 = 'data/dom1.jpg'
jpgname2 = 'data/dom2.jpg'
imgname1 = 'data/dom1.img'
imgname2 = 'data/dom2.img'

sift = cv2.xfeatures2d.SIFT_create()
matchRadio = 0.45

# FLANN 参数设计
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

# 灰度处理图像
img1 = cv2.imread(jpgname1)
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

# des是描述子
kp1, des1 = sift.detectAndCompute(img1, None)

img2 = cv2.imread(jpgname2)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
kp2, des2 = sift.detectAndCompute(img2, None)
# 水平拼接
hmerge = np.hstack((gray1, gray2))
# 拼接显示为gray
# cv2.imshow("gray", hmerge)
# cv2.waitKey(0)

# img3 = cv2.drawKeypoints(img1, kp1, img1, color=(255, 0, 255))
# img4 = cv2.drawKeypoints(img2, kp2, img2, color=(255, 0, 255))
# 水平拼接
# hmerge = np.hstack((img3, img4))
# 拼接显示为gray
# cv2.imshow("point", hmerge)
# cv2.waitKey(0)
matches = flann.knnMatch(des1, des2, k=2)
matchesMask = [[0, 0] for i in range(len(matches))]

good = []
# 输出结果集合，分别是[x1,y1,x2,y2]
results = []
for m, n in matches:
    if m.distance < matchRadio * n.distance:
        good.append([m])
        result_one = [kp1[m.queryIdx].pt[0], kp1[m.queryIdx].pt[1], kp2[m.trainIdx].pt[0],
                      kp2[m.trainIdx].pt[1]]
        results.append(result_one)
        print(result_one)

img5 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)

# 展示图片
cv2.namedWindow("enhanced", 0)
cv2.resizeWindow("enhanced", 1600, 900)
cv2.imshow('enhanced', img5)
# 等待按键按下
cv2.waitKey(0)
# 清除所有窗口
cv2.destroyAllWindows()
