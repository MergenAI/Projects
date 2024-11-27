import cv2
import numpy as np
import matplotlib.pyplot as plt
import keyboard




def a(x):
    pass

def kenar(resim,alt_1,alt_2):
    # 3 kanallı görseli tek kanala indirgeyip işlemden tasarruf etmek ve daha isabetli tespitler için
    gri = cv2.cvtColor(resim, cv2.COLOR_BGR2GRAY)
    # belirlenen matris, görseli tarar ve piksellerin ortalamasıyla değiştirir. Böylelikle sapmalar azalır
    bulanık = cv2.GaussianBlur(gri, (3, 3), 0)
    """
    1------------50,150
    2------------117,238
    3------------82,196
    4------------127,217
    """
    belirgin_kenar = cv2.Canny(bulanık,alt_1,alt_2)
    return belirgin_kenar
def görüntü_sınırla(resim):
    # cv,görseli yükseklik-genişlik-kanal olarak işler
    # 0,y-1100-y,634-340
    # 286-y,947-y,632-521
    # 0-y,1200-y,730-438

    yükseklik,genişlik=int(resim.shape[0]),int(resim.shape[1])
    üçgensel_alan=np.array([
        # [(700,yükseklik),(1250,yükseklik),(1030,148)]
        [(200,yükseklik),(500,585),(785,585),(1090,yükseklik)]
    ])
    # belirtilen matrisin birebir aynısı olacak şekilde yalnızca 0'dan oluşan yeni bir matris döndürür
    mask=np.zeros_like(resim)
    # atılan ilk parametreyi; ikinci parametrenin mevkilerini,üçüncü parametrenin değerine göre doldurur
    cv2.fillPoly(mask,üçgensel_alan,(255,0,0))
    ortak_küme=cv2.bitwise_and(resim,mask)
    return ortak_küme


def şerit_tamamla(resim,şeritler):
    şerit_görseli=np.zeros_like(resim)
    if şeritler is not None:
        for şerit in şeritler:
            try:
                x1,y1,x2,y2=şerit.reshape(4)
                cv2.line(şerit_görseli,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,255),5)
            except:
                pass
    şerit_bul1 = cv2.cvtColor(şerit_görseli, cv2.COLOR_BGR2GRAY)
#     # print(şerit_bul1.shape)
    thresh = cv2.threshold(şerit_bul1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    resim1=şerit_görseli.copy()
    # find all contours
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#     # print('len(contours):', len(contours))
    contour_area = []
    for c in contours:
        # print('area:', cv2.contourArea(c))
        contour_area.append((cv2.contourArea(c), c))

    # print('--- contour_area ---')
    # for item in contour_area:
        # print('contour_area:', item[0])

    # sort list with `(area, contour)` using only `area`
    contour_area = sorted(contour_area, key=lambda x: x[0], reverse=True)
#     # print('--- contour_area - sorted ---')
    # for item in contour_area:
#     #     print('contour_area:', item[0])
#     # print('--- two the biggest contours ---')
#     # print('p0:', contour_area[0][0])
#     # print('p1:', contour_area[1][0])
    try:
        coords1 = np.vstack([contour_area[0][1], contour_area[1][1]])
        coords = np.column_stack(np.where(thresh > 0))
        cv2.fillPoly(şerit_bul1, [coords1], (0, 255, 0))
        c2 = np.stack((coords[:, 1], coords[:, 0]), axis=-1)

        cv2.fillPoly(şerit_görseli, [c2], (255, 255, 0))
    except:
        pass
#     # print(resim1.shape,"resim1")
    # resim1=cv2.cvtColor(resim1,cv2.COLOR_GRAY2BGR)
    return şerit_görseli,resim1

def nokta_bul(resim,doğrular,oran):
    # try:
    #     m, c = doğrular
    # except TypeError:
    #     m, c = .001,0
    m, c = doğrular
    y1=resim.shape[0]
    """
    burada görsele göre ayarlama yapılabilir. Zira görselden görsele değişken bir boy oranı var ve bu oran,şeritlerin 
    uzunluğunu gösteriyor. Her iki şerit doğrusu kesişmemeli,kapalı alan oluşturmamalı 
    """
    y2=int(y1*oran)
    x1=int((y1-c)/m)
    x2=int((y2-c)/m)
    return np.array([x1, y1, x2, y2])
def ortalama_doğru_hesapla(resim,şeritler,oran):
    sağ_şerit_noktaları=[]
    sol_şerit_noktaları=[]
    if şeritler is not None:
        for şerit in şeritler:
            x1, y1, x2, y2 = şerit.reshape(4)
            # polinomik olarak verilen noktalar arasındaki bağlantıyı bulur.[x,y,derece]. eğim ve kesim noktasını döndürür
            params=np.polyfit((x1,x2),(y1,y2),1)
            m, c=params[0],params[1]
#             # print(params)
            if m<0:
                sağ_şerit_noktaları.append((m,c))
            else:
                sol_şerit_noktaları.append((m,c))
            # axis=0,x ekseni boyunca yani bütün sütunlar;axis=1,y ekseni boyunca yani bütün satırlar
        oran_1=oran
        # print(len(sağ_şerit_noktaları),"-----",len(sol_şerit_noktaları))



        if len(sol_şerit_noktaları)==len(sağ_şerit_noktaları)==0:
            return np.array([])
        elif len(sağ_şerit_noktaları)==0:
            sol_şerit_ortalaması=np.average(sol_şerit_noktaları,axis=0)
            sol_şerit=nokta_bul(resim,sol_şerit_ortalaması,oran)
            return np.array([sol_şerit])

        elif len(sol_şerit_noktaları)==0:
            sağ_şerit_ortalaması=np.average(sağ_şerit_noktaları,axis=0)
            sağ_şerit=nokta_bul(resim,sağ_şerit_ortalaması,oran)
            return np.array([sağ_şerit])


        sol_şerit_ortalaması=np.average(sol_şerit_noktaları,axis=0)
        sağ_şerit_ortalaması=np.average(sağ_şerit_noktaları,axis=0)
#         # print(sol_şerit_ortalaması,"----sol")
#         # print(sağ_şerit_ortalaması,"----sağ")
        sol_şerit=nokta_bul(resim,sol_şerit_ortalaması,oran)
        sağ_şerit=nokta_bul(resim,sağ_şerit_ortalaması,oran)
        return np.array([sol_şerit,sağ_şerit])

def görsel_denemesi(dosya):
    resim = dosya
    kopya=np.copy(resim)
    şerit=kenar(kopya,95,178)

    ortak_küme=görüntü_sınırla(şerit)
    # ortak_küme=şerit

    çizgiler=cv2.HoughLinesP(ortak_küme,2,np.pi/180,100,np.array([]),minLineLength=40,maxLineGap=150)
    ortalama=ortalama_doğru_hesapla(kopya,çizgiler,oran=.86)
    şerit_bul,resim1=şerit_tamamla(kopya,ortalama)
    # şerit_bul=şerit_tamamla(kopya,çizgiler)
    kesişim=cv2.addWeighted(kopya,.9,şerit_bul,1,1)
    ii=cv2.cvtColor(ortak_küme,cv2.COLOR_GRAY2BGR)
#     # print(ortak_küme.shape)
#     # print(şerit_bul.shape)
#     # print(ii.shape)
    cv2.namedWindow("pencere",cv2.WINDOW_NORMAL)
    toplu=np.vstack((ii,şerit_bul,resim1,kesişim))

    cv2.imshow("pencere",toplu)
    # cv2.imshow("pencere1",şerit_bul)
    cv2.waitKey(0)

    cv2.imshow("pencere2",şerit_bul)

"""
To stack vertically (img1 over img2):

vis = np.concatenate((img1, img2), axis=0)
To stack horizontally (img1 to the left of img2):

vis = np.concatenate((img1, img2), axis=1)
"""

def görsel_al(dosya,kare):
    video=cv2.VideoCapture(dosya)
    video.set(1,kare)
    _,kopya=video.read()
    total_frames=kopya
    cv2.imshow("pencere", total_frames)
    cv2.imwrite("test4.png",total_frames)
    cv2.waitKey(0)
def piksel_tespit_et():
    a=plt.imread("üçüncü.png")
    plt.imshow(a)
    plt.show()
def video_kaydet():
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('output1.avi', fourcc, 20.0, (1280,720))
    return out


#https://www.kaggle.com/soumya044/lane-line-detection
def video_denemesi():
    ekran=0
    video=cv2.VideoCapture(dosya_)
    cv2.namedWindow("pencere",cv2.WINDOW_NORMAL)
    out = video_kaydet()
    devam = True
    while video.isOpened() and devam:
        _,kopya=video.read()
        if kopya is not None:

            şerit = kenar(kopya,17,189)
            ortak_küme = görüntü_sınırla(şerit)
            çizgiler = cv2.HoughLinesP(ortak_küme, 2, np.pi / 180, 100, np.array([]), minLineLength=1, maxLineGap=1e20)
            ortalama = ortalama_doğru_hesapla(kopya, çizgiler,.85)
            şerit_bul,ii = şerit_tamamla(kopya, ortalama)
            # i=cv2.cvtColor(şerit,cv2.COLOR_GRAY2BGR)
            kesişim = cv2.addWeighted(kopya, .9, şerit_bul, 1, 1)
            ii = cv2.cvtColor(ortak_küme, cv2.COLOR_GRAY2BGR)
            şerit = cv2.cvtColor(şerit, cv2.COLOR_GRAY2BGR)
            for i in çizgiler:
                x1, y1, x2, y2 = i.reshape(4)

                kopya=cv2.line(kopya,(x1,x2),(y1,y2),color=(0,0,0),thickness=10)
            toplu = np.vstack((şerit, şerit_bul, kopya, ii))

            out.write(kesişim)

            # toplu=np.vstack((kesişim,i,ii))
#             # print(ekran)
            cv2.imshow("pencere", toplu)
            # print(kesişim.shape)
            # cv2.imshow("pencere1", ortak_küme)
            # cv2.imshow("pencere2", şerit_bul)
            # ekran+=1
            key=cv2.waitKey(25)
            if key==ord("q"):
                kopya.release()
                cv2.destroyAllWindows()
                break
            if key==ord("a"):
                print(1)
                devam=False
                # cv2.waitKey(-1)


    # #
def alt_üst_ayarla(dosya):
    """
    araba.png için kırpma noktaları=(399,279)
    """

    cv2.namedWindow("pencere",cv2.WINDOW_NORMAL)

    cv2.createTrackbar("altSınır","pencere",127,255,a)
    cv2.createTrackbar("üstSınır","pencere",217,255,a)
    cv2.createTrackbar("kesim","pencere",0,100,a)
    matris=7
    while True:
        resim=dosya
        kopya=resim.copy()
        alt_sınır=cv2.getTrackbarPos("altSınır","pencere")
        üst_sınır=cv2.getTrackbarPos("üstSınır","pencere")
        kesim=cv2.getTrackbarPos("kesim","pencere")
        resim=cv2.GaussianBlur(resim,(matris,matris),1)
        resim=cv2.Canny(resim, alt_sınır, üst_sınır)

        ortak_küme = görüntü_sınırla(resim)

        çizgiler = cv2.HoughLinesP(ortak_küme, 3, np.pi / 180, 25, np.array([]), minLineLength=40, maxLineGap=20000)
        # print(çizgiler)
        ortalama = ortalama_doğru_hesapla(kopya, çizgiler,kesim/100)
        şerit_bul, resim1 = şerit_tamamla(kopya, ortalama)
        # şerit_bul=şerit_tamamla(kopya,çizgiler)
        kesişim = cv2.addWeighted(kopya, .9, şerit_bul, 1, 1)
        ii = cv2.cvtColor(ortak_küme, cv2.COLOR_GRAY2BGR)
#         # print(ortak_küme.shape)
#         # print(şerit_bul.shape)
#         # print(ii.shape)
        toplu = np.vstack((ii, şerit_bul, kesişim, resim1))
        # ortak_küme=görüntü_sınırla(resim)
        # çizgiler=cv2.HoughLinesP(ortak_küme,2,np.pi/180,100,np.array([]),minLineLength=0,maxLineGap=150)
        # ortalama=ortalama_doğru_hesapla(kopya,çizgiler)
        # thresh = cv2.threshold(resim, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        #
        # contours, hierarchy = cv2.findContours(resim, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#         # print('len(contours):', len(contours))
        # contour_area = []
        # for c in contours:
#         #     print('area:', cv2.contourArea(c))
        #     contour_area.append((cv2.contourArea(c), c))
        # şerit_bul=şerit_tamamla(kopya,ortalama)
        # kesişim=cv2.addWeighted(kopya,.8,şerit_bul,1,1)
        #
        cv2.imshow("pencere", toplu)
        # cv2.imshow("pencere2", kesişim)
        # cv2.imshow("pencere1", şerit_bul)
        if cv2.waitKey(1) == ord("q"):
            cv2.destroyAllWindows()
            break



def eğim(x1, y1, x2, y2 ):
    return (y2-y1)/(x2-x1)

def sabit(x1, y1, x2, y2):
    m = eğim(x1, y1, x2, y2)
    c=(m*(0-x1))+y1
    return c

def doğru(x1, y1, x2, y2):
    m=eğim(x1, y1, x2, y2)
    c = sabit(x1, y1, x2, y2)
    y=x1*m+c
    return int(y)
dosya_="./F/test4.mp4"
dosya=cv2.imread("test4.png")
şerit = kenar(dosya, 17, 120)
ortak_küme = görüntü_sınırla(şerit)
çizgiler = cv2.HoughLinesP(ortak_küme, 2, np.pi / 180, 100, np.array([]), minLineLength=1, maxLineGap=1e20)
sağ,sol=[],[]
print(çizgiler.shape)
for i in çizgiler:
    x1, y1, x2, y2 = i.reshape(4)
    m,c=np.polyfit((x1,x2),(y1,y2),1)
    if m<0:
        sağ.append([x1, y1, x2, y2])
    else:
        sol.append([x1, y1, x2, y2])

print(len(sağ))
print(len(sol))
for s in sağ:
    sağ.sort(key=lambda x:x[1])
for ii in sol:
    sol.sort(key=lambda x1:x1[1])
for i in range(len(sol)-1):
    x1, y1, x2, y2 = sol[i]
    x_1, y_1, x_2, y_2 = sol[i+1]
    y=doğru(x1, y1, x2, y2)
    y_=doğru(x_1, y_1, x_2, y_2)
    dosya=cv2.line(dosya,(x1,y),(x_1,y_),color=(0,0,0),thickness=10)

cv2.imshow("1",dosya)
cv2.waitKey(0)


# alt_üst_ayarla(dosya)
# video_denemesi()
# alt_üst_ayarla(dosya)
# görsel_denemesi(dosya)
# görsel_al(dosya_,250)