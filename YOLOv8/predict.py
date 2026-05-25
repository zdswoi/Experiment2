from ultralytics import YOLO
import numpy
import os

# Load a COCO-pretrained YOLOv8n model
model = YOLO("yolov8n.pt",task="detect")


# Run inference with the YOLOv8n model on the 'bus.jpg' image
results = model(source="E:/yanyishang/shiyan2/ultralytics-main-2/ultralytics/assets/bus.jpg",save=True,conf=0.7,iou=0.7)
# 如果视频或者文件夹的话，可以show=True，这样可以把检测的结果展示出来，更有利于观察
# save=true(只能是'save=True' or 'save=False')是保存图片会将图片保存到runs\detect\predict中，
# conf是置信度，iou其值可以理解为是一个与conf相反的东西，越大框会越多，与NMS有关
print(results[0].boxes.xywh)
# 其含有xywh（中心点的坐标，高和宽），xyxy（左上角，右下角），但是现在还是一个tensor张量的形式要进行转换
print(results[0].boxes.xywh.cpu().numpy())

# 3. 核心：获取YOLO自动生成的输出目录路径
# 从results中提取保存路径（与预测图片同目录）
save_dir = results[0].save_dir  # 关键！获取当前推理的输出目录（如runs/detect/predict3）

# 4. 提取检测框数据（xywh：中心点x, 中心点y, 宽度w, 高度h）
boxes_xywh = results[0].boxes.xywh.cpu().numpy()
cls_ids = results[0].boxes.cls.cpu().numpy()  # 类别ID（可选）
# confidences = results[0].boxes.conf.cpu().numpy()  # 置信度（可选）

# 2. 待预测的图片路径
img_path = r'E:/yanyishang/shiyan2/ultralytics-main-2/ultralytics/assets/bus.jpg'
# 4. 获取save=True对应的输出目录
save_dir = results[0].save_dir
# 5. 拼接TXT文件路径
img_name = os.path.basename(img_path)
txt_name = os.path.splitext(img_name)[0] + '.txt'
txt_path = os.path.join(save_dir, txt_name)

# 6. 提取预测结果并写入TXT（用str.format()替代f-string）
result = results[0]
with open(txt_path, 'w', encoding='utf-8') as f:
    for box in result.boxes:
        # 提取核心信息
        cls_id = int(box.cls.cpu().numpy())
        cls_name = result.names[cls_id]
        conf = box.conf.cpu().numpy()[0]
        xywh = box.xywh.cpu().numpy()[0]  # xyxy格式坐标

        # 核心修复：用str.format()替代f-string，兼容所有Python3版本
        line = "{}\t{}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\n".format(
            cls_id, cls_name,
            xywh[0], xywh[1], xywh[2], xywh[3]
        )
        f.write(line)

# 打印保存信息（同样用format替代f-string）
print("预测结果已保存到：{}".format(txt_path))
print("预测图片保存到：{}/{}".format(save_dir, img_name))