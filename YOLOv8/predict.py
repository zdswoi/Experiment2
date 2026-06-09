from ultralytics import YOLO
import numpy as np
import os

# ========== 1. 修正：加载你自己训练好的best.pt权重 ==========
model_path = r"E:/yanyishang/shiyan2/ultralytics-main-2/runs/detect/train8/weights/best.pt"
model = YOLO(model_path)

img_path = r'E:/研一上/5 抗生素检测/实验2/摘要版图片-1.png'

# 推理预测
results = model(
    source=img_path,
    save=True,    # 保存带框效果图
    conf=0.2,     # 置信阈值
    iou=0.2       # NMS交并比阈值
)

result = results[0]
# 打印原始张量坐标
print("张量格式xywh：")
print(result.boxes.xywh)
print("numpy数组xywh：")
print(result.boxes.xywh.cpu().numpy())

# 获取保存文件夹
save_dir = result.save_dir

# 拆分图片名称，生成txt文件名
img_name = os.path.basename(img_path)
txt_name = os.path.splitext(img_name)[0] + '.txt'
txt_path = os.path.join(save_dir, txt_name)

# ========== 2. 写入txt文件 ==========
with open(txt_path, 'w', encoding='utf-8') as f:
    for box in result.boxes:
        cls_id = int(box.cls.cpu().numpy())
        cls_name = result.names[cls_id]
        conf = float(box.conf.cpu().numpy()[0])
        xywh = box.xywh.cpu().numpy()[0]  # 中心点x,y 宽w 高h

        # 写入内容：类别ID、类别名、置信度、x、y、w、h
        line = "{}\t{}\t{:.3f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\n".format(
            cls_id, cls_name, conf,
            xywh[0], xywh[1], xywh[2], xywh[3]
        )
        f.write(line)

# 输出提示
print("检测结果txt保存路径：{}".format(txt_path))
print("标记效果图保存路径：{}/{}".format(save_dir, img_name))
