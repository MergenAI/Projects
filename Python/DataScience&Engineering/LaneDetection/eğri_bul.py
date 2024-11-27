from numpy import sin,tan
from numpy import sqrt
from numpy import arange
from scipy.optimize import curve_fit
from matplotlib import pyplot
import cv2
import numpy as np

# define the true objective function
def objective(x, a, b, c, d):
    return a * sin(b - x) + c * x ** 2 + d
    # return a * tan(b - x) + c * x ** 2 + d
def a(x):
    pass

def eğri_çiz(şeritler):
    # choose the input and output variables
    # x, y = [1,3,5,6,8,10,11,12,13],[1,2,4,5,4,8,10,11,9]
    # curve fit
    x,y=[],[]
    if şeritler is not None:
        for şerit in şeritler:
            x1, x2,y1,y2 = şerit.reshape(4)
            x.append(x1)
            x.append(x2)
            y.append(y1)
            y.append(y2)
    popt, _ = curve_fit(objective, x, y)
    # summarize the parameter values
    a, b, c, d = popt
    # print(popt)
    # plot input vs output
    pyplot.scatter(x, y)
    # # define a sequence of inputs between the smallest and largest known inputs
    x_line = arange(min(x), max(x), 1)
    # # calculate the output for the range
    y_line = objective(x_line, a, b, c, d)
    # pyplot.plot(x_line, y_line, '--', color='red')
    # pyplot.show()
    return np.array([x_line,y_line])
    # # create a line plot for the mapping function

def kenar(dosya):
    cv2.namedWindow("pencere", cv2.WINDOW_NORMAL)
    cv2.createTrackbar("altSınır", "pencere", 127, 255, a)
    cv2.createTrackbar("üstSınır", "pencere", 217, 255, a)
    while True:
        resim = dosya
        alt_sınır = cv2.getTrackbarPos("altSınır", "pencere")
        üst_sınır = cv2.getTrackbarPos("üstSınır", "pencere")
        resim = cv2.GaussianBlur(resim, (3, 3), 0)
        resim = cv2.Canny(resim, alt_sınır, üst_sınır)
        çizgiler = cv2.HoughLinesP(resim, 1, np.pi / 180, 25, np.array([]), minLineLength=0, maxLineGap=20000)
        # print(çizgiler)
        kavis_denklemi=eğri_çiz(çizgiler)
        print(kavis_denklemi)
        # print(kavis_denklemi)
        # img_dilate = cv2.dilate(resim, None, iterations=1)
        # img_erode = cv2.erode(img_dilate, None, iterations=1)
        # # thresh = cv2.threshold(resim, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        #
        # contours, hierarchy = cv2.findContours(img_erode, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        #
        # contour_area = []
        #
        # for c in contours:
        #     # print('area:', cv2.contourArea(c))
        #     contour_area.append((cv2.contourArea(c), c))
        #
        # for cnt in contours:
        #     cv2.drawContours(img_erode, cnt, -1, 0, -1)
        # contour_area = sorted(contour_area, key=lambda x: x[0], reverse=True)
        # # print('p0:', contour_area[0][0])
        # # print('p1:', contour_area[1][0])
        # coords1 = np.vstack([contour_area[0][1], contour_area[1][1]])
        # print(coords1)
        #
        # cv2.fillPoly(resim, [coords1], (255, 255, 255))
        # coords = np.column_stack(np.where(thresh > 0))
        # thresh = cv2.threshold(resim, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        # c2=np.stack((coords[:,1], coords[:,0]), axis=-1)
        #
        # cv2.fillPoly(resim, [c2], (255, 255, 255))
        resim=cv2.cvtColor(resim,cv2.COLOR_GRAY2BGR)
        img_erode=cv2.cvtColor(img_erode,cv2.COLOR_GRAY2BGR)
        ortak=np.vstack((resim,dosya,img_erode))
        cv2.imshow("pencere",ortak)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


dosya=cv2.imread("F/Adsiz.png")
kenar(dosya)


# def denklem(katsayı,dizi):
#     return np.array([(katsayı*eleman**2) for eleman in dizi])
#
# cv2.namedWindow("1",cv2.WINDOW_NORMAL)
# x=[0,1,2,3,4,5]
# y1=denklem(1,x)
# y2=denklem(4,x)
# print(y1)
# print(y2)
# nokta=[]
# for i in range(len(x)):
#     nokta.append([x[i],y1[i]])
#     nokta.append([x[i],y2[i]])
# nokta=np.array(nokta)
# print(nokta)
# görsel=np.zeros(shape=(100,100))
# pts = nokta
# pts = pts.reshape((-1,1,2))
# cv2.polylines(görsel,[pts],True,(255,255,255))
# cv2.fillPoly(görsel, [pts], (255, 255, 255))
# cv2.imshow("1",görsel)
# cv2.waitKey(0)

