



def toplamları_bul(sayı):
    if sayı>0:
       sayı+toplamları_bul(sayı - 1)
    else:
        return 0



print(toplamları_bul(sayı=10))