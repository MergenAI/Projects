import cv2
import os
import time
import PIL.ImageGrab as f
import win32gui
import numpy as np
dosya_yolu="D:\TF_OD_API\Tensorflow\workspace\images"
sayaç=0
dosya_adı="ordek_{}.jpg"
bitir=time.time()+10
p=win32gui.FindWindow(None,"DuckHuntJS - Opera")
d=win32gui.GetWindowRect(p)
x1,y1,x2,y2=d
def winEnumHandler( hwnd, ctx ):
    if win32gui.IsWindowVisible( hwnd ):
        print (hex(hwnd), win32gui.GetWindowText( hwnd ))

win32gui.EnumWindows( winEnumHandler, None )


for i in range(3)[::-1]:
    print(i)
    time.sleep(1)
while time.time()<bitir:
    g=f.grab(bbox=d)
    g=np.array(g)
    g=cv2.resize(g,(800,600))
    g=cv2.cvtColor(g,cv2.COLOR_BGR2RGB)
    cv2.imwrite(filename=dosya_yolu+"\\"+dosya_adı.format(sayaç),img=g)
    sayaç+=1
#
# for i in os.listdir(dosya_yolu):
#     print(i)
#     ad="ördek_{}.jpg".format(sayaç)
#     sayaç+=1
#     os.rename(dosya_yolu+"\\"+i,dosya_yolu+"\\"+ad)
