from ultralytics import YOLO
from ultralytics.nn.tasks import yaml_model_load

print(yaml_model_load(r'D:\Research\msc_research\yolo\yolo12_forked\yolov12-pvelad\ultralytics\cfg\models\v12\yolov12kan.yaml'))
# model = YOLO(r'D:\Research\msc_research\yolo\yolo12_forked\yolov12-pvelad\ultralytics\cfg\models\v12\yolov12kan.yaml')