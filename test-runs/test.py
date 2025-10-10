from ultralytics import YOLO
from ultralytics.nn.tasks import yaml_model_load

# print(yaml_model_load(r'D:\Research\msc_research\yolo\yolov12-pvelad\ultralytics\cfg\models\custom_model.yaml'))
model = YOLO(r'D:\Research\msc_research\yolo\yolov12-pvelad\ultralytics\cfg\models\custom_model.yaml',task="detect")