import ctypes
import time

import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import win32api as w32
import win32gui
from PIL import ImageGrab
# window_handle = w32.FindWindow(None, "Diablo II")
# window_rect   = w32.GetWindowRect(window_handle)

#
# def winEnumHandler( hwnd, ctx ):
#     if win32gui.IsWindowVisible( hwnd ):
#         print (hex(hwnd), win32gui.GetWindowText( hwnd ))
# win32gui.EnumWindows( winEnumHandler, None )


def ekran_görüntüsü_al():
    for i in range(5)[::-1]:
        print(i)
        time.sleep(1)
    x1, y1, x2, y2 = win32gui.GetWindowRect(win32gui.FindWindow(None, "Grand Theft Auto V"))
    ekran_alıntısı = ImageGrab.grab(bbox=(x1, y1, x2, y2))
    plt.imshow(ekran_alıntısı)
    plt.show()
    ekran_alıntısı = np.array(ekran_alıntısı)
    ekran_alıntısı = cv.cvtColor(ekran_alıntısı, cv.COLOR_BGR2RGB)
    cv.imwrite("geta5-1.png", ekran_alıntısı)

def koordinat_gör(dosya):
    ekran=dosya
    plt.imshow(ekran)
    plt.show()

dosya=plt.imread("geta5-1.png")
koordinat_gör(dosya)
# ekran_görüntüsü_al()