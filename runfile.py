from test import *
modelURL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz'
# modelURL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d4_coco17_tpu-32.tar.gz'
classFile = 'coco.names'
imagePath = 'media/1.jpg'
# vidPath = 'media/v3.mp4'

# vidPath = 0
threshold = 0.5

detector = Detector()
detector.readClasses(classFile)
detector.downLoadModel(modelURL)
detector.loadModel()
detector.predictImage(imagePath, threshold)
# detector.predictVid(vidPath, threshold)