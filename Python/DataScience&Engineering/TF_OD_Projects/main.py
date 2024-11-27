import os
import time
import tensorflow as tf
import win32api
import win32con
import win32gui
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format
import wget
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
import numpy as np
import cv2
import PIL.ImageGrab as ig
import pygetwindow as pg
import conf

CONFIG_PATH=conf.CONFIG_PATH
CHECKPOINT_PATH=conf.CHECKPOINT_PATH
ANNOTATION_PATH=conf.ANNOTATION_PATH
IMAGE_PATH=conf.IMAGE_PATH

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(CONFIG_PATH)
detection_model = model_builder.build(model_config=configs['model'], is_training=False)
#
# # Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(CHECKPOINT_PATH, 'ckpt-6')).expect_partial()



def önişlemler():
    pencere = pg.getWindowsWithTitle('DuckHuntJS - Opera')[0]
    resim = ig.grab(bbox=(pencere.topleft.x, pencere.topleft.y, pencere.bottomright.x, pencere.bottomright.y))
    resim = np.array(resim)
    # resim=cv2.cvtColor(resim,cv2.COLOR_BGR2RGB)
    width = int(resim.shape[0])
    height = int(resim.shape[1])
    image_np = np.array(resim)
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    pencere_x,pencere_y=pencere.topleft[0], pencere.topleft[1]
    return width,height,input_tensor,image_np,pencere_x,pencere_y

@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections




category_index = label_map_util.create_category_index_from_labelmap(ANNOTATION_PATH+'/label_map.pbtxt')
def tahmin_et():
    width, height, input_tensor, image_np,pencere_x,pencere_y = önişlemler()

    detections = detect_fn(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'],
        detections['detection_classes'] + label_id_offset,
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=5,
        min_score_thresh=.5,
        agnostic_mode=False)

    detections = detect_fn(input_tensor)

    # This is the way I'm getting my coordinates
    boxes = detections['detection_boxes'].numpy()[0]
    # print("boxes",boxes)
    # print("boxes.shape",boxes.shape[0])
    # print("boxes.shape",boxes.shape)
    # get all boxes from an array
    max_boxes_to_draw = boxes.shape[0]
    # print("max_boxes_to_draw",max_boxes_to_draw)
    # get scores to get a threshold
    scores = detections['detection_scores'][0]
    # print("scores",scores)
    # this is set as a default but feel free to adjust it to your needs
    min_score_thresh = .5
    # # iterate over all objects found
    coordinates = []

    for i in range(min(max_boxes_to_draw, boxes.shape[0])):
        if scores[i] > min_score_thresh:
            class_id = int(detections['detection_classes'][0][i] + 1)
            coordinates.append([boxes[i], class_id])
    return coordinates,width, height,pencere_x,pencere_y,image_np_with_detections

p1=cv2.namedWindow("pencere1",cv2.WINDOW_NORMAL)
ilk_zaman=time.time()
while True:

    coordinates,width, height,pencere_x,pencere_y,image_np_with_detections=tahmin_et()
    # son_zaman=time.time()
    mevki=[]
    if not len(coordinates)==0:
        for i,sıra in coordinates:
            if sıra==1:
                ymin,ymax=i[0]*width,i[2]*width
                xmin,xmax=i[1]*height,i[3]*height
                yort,xort=int((ymax+ymin)/2),int((xmax+xmin)/2)

                # image_np_with_detections=cv2.putText(image_np_with_detections,str(1/(son_zaman-ilk_zaman)),(0,150),cv2.FONT_HERSHEY_COMPLEX,1, (255, 255,255), 2)
                image_np_with_detections=cv2.circle(image_np_with_detections,(xort,yort),5,(255,255,255),cv2.FILLED,5)
                print(f"{xmin,ymin,xmax,ymax}",f"{xort,yort}")

                pencere1=win32gui.FindWindow(None,p1)
                pencere1=win32gui.GetWindowRect(pencere1)
                x,y=pencere_x,pencere_y
                mevki.append([int(xort + x), int(yort + y)])
                if not len(mevki)==0:
                    print(win32gui.GetCursorPos(),"---------------",xort,yort,"---------------",x,y,"---------------",mevki[0][0],mevki[0][1])
                    win32api.SetCursorPos((mevki[0][0],mevki[0][1]))
                    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0)
                    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0)
                    mevki=[]
    # ilk_zaman=son_zaman
    image_np_with_detections=cv2.cvtColor(image_np_with_detections,cv2.COLOR_BGR2RGB)
    cv2.imshow("pencere1", image_np_with_detections)
    if cv2.waitKey(25)&0xFF==ord("q"):
        break


# https://duckhuntjs.com
# # import cv2,os
# # dosya="D:\TF_OD_API\Tensorflow\workspace\images"
# # for i in os.listdir(dosya):
# #     if i.endswith(".jpg"):
# #         print(i)
# #         gö=cv2.imread(os.path.join(dosya,i))
# #         gö=cv2.resize(gö,(320,320))
# #         cv2.imwrite(str(i[:-4]+"yeni"+i[-4:]),gö)

#win32api.GetKeyState(0x01) sol düğme 0-1  0x02 sağ düğme

