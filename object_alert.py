#!/usr/bin/python3.7

#----------------------------------------------
#--- Author         : Irfan Ahmad
#--- Mail           : irfanibnuahmad@gmail.com
#--- Date           : 5th July 2020
#----------------------------------------------

# Object detection imports
from utils import backbone
from api import object_alert_api as oa_api

# By default I use an "SSD with Mobilenet" model here. See the detection model zoo (https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.
detection_graph, category_index = backbone.set_model('ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03', 'mscoco_label_map.pbtxt') # 26 ms
# detection_graph, category_index = backbone.set_model('ssd_mobilenet_v1_coco_2018_01_28', 'mscoco_label_map.pbtxt') # 30 ms
# detection_graph, category_index = backbone.set_model('faster_rcnn_inception_v2_coco_2018_01_28', 'mscoco_label_map.pbtxt') # 58 ms

input_video_path = './input/'

##########################################################
##########################################################

input_video = 'Video Danger Alert.mp4'
start_point = (224, 52)
end_point = (416, 281)

##########################################################
##########################################################

oa_api.object_alert(input_video_path + input_video, detection_graph, category_index, start_point, end_point)
