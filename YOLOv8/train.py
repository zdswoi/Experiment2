
from ultralytics import YOLO

model = YOLO("./yolov8n.pt") 

model.train(data="data.yaml",workers=0,epochs=10,batch=16,imgsz=640) 
# yolov5===data="data.yaml",workers=4,epochs=100,batch=2//////workers如果是wins系统设置成1就可以
