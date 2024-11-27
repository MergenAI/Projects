import os
import xml.etree.ElementTree as et
import numpy as np
dosya_yolu="D:\TF_OD_API\Tensorflow\workspace\images\\test"

# for i in os.listdir(dosya_yolu):
#     ad="o"+i[1:]
#     print(ad)
#     os.rename(dosya_yolu+"\\"+i,dosya_yolu+"\\"+ad)
b=[]
for i in os.listdir(dosya_yolu):
    if i.endswith(".xml"):
        t=et.parse(os.path.join(dosya_yolu,i))
        r=t.getroot()
        for path in r.iter("path"):
            a=path.text
            b.append(a[47:52])
b=np.array(b)
print(np.unique(b))



# jpg_=[]
# xml_=[]
#
# for i in os.listdir(dosya_yolu):
#     if i.endswith(".jpg"):
#         jpg_.append(i)
#     elif i.endswith(".xml"):
#         xml_.append(i)
#
#
#
# for i in xml_:
#     ad=i[:-3]+"jpg"
#     if ad in jpg_:
#         jpg_.remove(ad)
# for i in jpg_:
#     os.remove(os.path.join(dosya_yolu,i))
