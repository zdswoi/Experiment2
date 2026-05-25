
# YOLOv8，YOLOv11，均以可以跑通，可以直接用这两个中的一个进行测试（可以这个实验用8，下个用11，也可以设计成对比的形式，对比性能，效率等等，5和8，8和11，5和8和11）
# YOLOv1-12种模型，还有最近最新出的YOLO26，其中YOLO26已找到可以参照的视频，并且一定可以跑通，可以等之后用这个模型试试跑通
# YOLO26模型也已经跑通了，可以留到下一个实验进行使用
# 其中如果不直接使用检测，而是转化成定位再找颜色，关于定位的代码也已经写好了，选点提取颜色可以用之前的
from ultralytics import YOLO

model = YOLO("./yolov8n.pt") # yolov8s收敛速度太快了，而且和yolov5n没有对比性，还是换成yolov8n跑吧，如果用S跑感觉50轮就非常够了，直接用s太快收敛了，所以还是用n

model.train(data="data-1.yaml",workers=0,epochs=10,batch=16,imgsz=640) # 100轮感觉完全足够了，感觉50轮就完全足够了，但是如果用5来跑的话，肯定还是需要100轮才够
# yolov5===data="data-1.yaml",workers=4,epochs=100,batch=2//////workers如果是wins系统设置成1就可以

# yolo task=detect mode=train model=./yolov8s.pt data=data-1.yaml epochs=3 workers=1 batch=16
# 上述是利用终端可以运行的代码，如果你想要直接让代码版的YOLOv8运行起来则需要将workers设置为0
# 终端里workers要等于1，代码里workers要等于0
# 实际检测中就是会存在各种各样的问题存在，所以采集数据也可以保留这种随机性，不是需要所有的图像都是一样的
# 可以有清楚的不清楚的，也可以有正的斜的
# 目前的想法是为了减少问题，可以学一些文献里一样，不放实验数据，或者不放输入数据，只放结果的内容，其中放在结果里的内容，可以选择一些拍的清楚的进行使用
# Ultralytics YOLOv8.2.48 🚀 Python-3.8.20 torch-2.2.0+cu121 CUDA:0 (NVIDIA GeForce RTX 3060 Laptop GPU, 6144MiB)
# engine\trainer: task=detect, mode=train, model=./yolov8n.pt, data=data-1.yaml, epochs=100, time=None, patience=100, batch=16, imgsz=640, save=True, save_period=-1, cache=False, device=None, workers=0, project=None, name=train8, exist_ok=False, pretrained=True, optimizer=auto, verbose=True, seed=0, deterministic=True, single_cls=False, rect=False, cos_lr=False, close_mosaic=10, resume=False, amp=True, fraction=1.0, profile=False, freeze=None, multi_scale=False, overlap_mask=True, mask_ratio=4, dropout=0.0, val=True, split=val, save_json=False, save_hybrid=False, conf=None, iou=0.7, max_det=300, half=False, dnn=False, plots=True, source=None, vid_stride=1, stream_buffer=False, visualize=False, augment=False, agnostic_nms=False, classes=None, retina_masks=False, embed=None, show=False, save_frames=False, save_txt=False, save_conf=False, save_crop=False, show_labels=True, show_conf=True, show_boxes=True, line_width=None, format=torchscript, keras=False, optimize=False, int8=False, dynamic=False, simplify=False, opset=None, workspace=4, nms=False, lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1, box=7.5, cls=0.5, dfl=1.5, pose=12.0, kobj=1.0, label_smoothing=0.0, nbs=64, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, bgr=0.0, mosaic=1.0, mixup=0.0, copy_paste=0.0, auto_augment=randaugment, erasing=0.4, crop_fraction=1.0, cfg=None, tracker=botsort.yaml, save_dir=runs\detect\train8
# Overriding model.yaml nc=80 with nc=82

# 可以把图像的数据精美一下，然后再重新跑个十轮得到一个新的train-batch0的数据值