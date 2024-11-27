import cv2
import numpy as np
görsel=cv2.imread("test4.png")
g=görsel.copy()
g1=görsel.copy()
görsel=cv2.cvtColor(görsel,cv2.COLOR_BGR2GRAY)

görsel1=cv2.GaussianBlur(görsel,(3,3),0)
görsel2=cv2.Canny(görsel1,0,255)
# print(görsel.shape)
# print(görsel1.shape)
# print(görsel2.shape)
# c=360/1280
# görsel=cv2.resize(görsel,(int(görsel.shape[0]*c),int(görsel.shape[1]*c)))
# görsel1=cv2.resize(görsel1,(int(görsel1.shape[0]*c),int(görsel1.shape[1]*c)))
# görsel2=cv2.resize(görsel2,(int(görsel2.shape[0]*c),int(görsel2.shape[1]*c)))
# görsel3=np.vstack((görsel2,görsel1,görsel))
#
# cv2.imshow("",görsel3)
# cv2.waitKey(0)
imshape = görsel2.shape
vertices = np.array([[(389,imshape[0]),(572, 550), (685, 550), (895,imshape[0])]], dtype=np.int32)
mask = np.zeros_like(görsel2)
cv2.fillPoly(mask, vertices,255)
masked_edges = cv2.bitwise_and(görsel2, mask)
rho = 2  # distance resolution in pixels of the Hough grid
theta = np.pi / 180  # angular resolution in radians of the Hough grid
threshold = 15  # minimum number of votes (intersections in Hough grid cell)
min_line_length = 40  # minimum number of pixels making up a line
max_line_gap = 30  # maximum gap in pixels between connectable line segments

lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
                                min_line_length, max_line_gap)
for line in lines:
    for x1, y1, x2, y2 in line:
        cv2.line(g, (x1, y1), (x2, y2), (255, 0, 0), 10)
lines_edges = cv2.addWeighted(g1, 0.8, g, 1, 0)
cv2.imshow("",lines_edges)
cv2.waitKey(0)
