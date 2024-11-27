


labels = [{'name': 'ordek', 'id': 1},{'name': 'agac', 'id': 2},{'name': 'kopek', 'id': 3},{"name":"olu_ordek", 'id':4}]

# https://github.com/nicknochnack/RealTimeObjectDetection/blob/main/Tutorial.ipynb
def lab_map_yarat():
    with open(ANNOTATION_PATH + '\label_map.pbtxt', 'w') as f:
        for label in labels:
            f.write('item { \n')
            f.write('\tname:\'{}\'\n'.format(label['name']))
            f.write('\tid:{}\n'.format(label['id']))
            f.write('}\n')


lab_map_yarat()
os.mkdir('Tensorflow\workspace\models\\'+CUSTOM_MODEL_NAME)

yol="D:\TF_OD_API\Tensorflow\workspace\pre-trained-models"
# wget.download('http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d7_coco17_tpu-32.tar.gz',out=yol)
wget.download('http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz',out=yol)
