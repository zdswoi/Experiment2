import os
import random
import shutil
from pathlib import Path

# ===================== 核心配置参数（请根据你的实际路径修改）=====================
# 标签文件根路径（YOLO格式.txt，与图片同名）
LABEL_ROOT = "VOCdevkit/VOC2007/YOLOLabels"
# 图片文件根路径（支持jpg/jpeg/png/bmp等格式）
IMAGE_ROOT = "VOCdevkit/VOC2007/JPEGImages"
# 训练集/测试集划分比例（7:3）
TRAIN_RATIO = 0.7
# 划分后文件保存的根目录（会自动创建train/val子目录）
OUTPUT_ROOT = "VOCdevkit"
# 随机种子（固定种子保证划分结果可复现）
RANDOM_SEED = 42
# 支持的图片格式（无需修改）
SUPPORTED_IMAGE_EXT = [".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG"]


# ===================== 工具函数：检查路径是否存在 =====================
def check_path_exists(path, name):
    """检查路径是否存在，不存在则报错"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"【错误】{name}路径不存在：{path}")
    return True


# ===================== 核心函数：划分并复制文件 =====================
def split_and_copy_dataset():
    try:
        # 1. 基础检查：确认原始路径存在
        check_path_exists(LABEL_ROOT, "标签文件")
        check_path_exists(IMAGE_ROOT, "图片文件")

        # 2. 创建输出目录（train/val下分别存放images和labels）
        train_image_dir = os.path.join(OUTPUT_ROOT, "train", "images")
        train_label_dir = os.path.join(OUTPUT_ROOT, "train", "labels")
        val_image_dir = os.path.join(OUTPUT_ROOT, "val", "images")
        val_label_dir = os.path.join(OUTPUT_ROOT, "val", "labels")

        # 递归创建目录（已存在则忽略）
        for dir_path in [train_image_dir, train_label_dir, val_image_dir, val_label_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"【信息】输出目录已创建/确认：{OUTPUT_ROOT}")

        # 3. 获取所有匹配的图片-标签文件对（仅保留有标签的图片）
        matched_files = []
        all_image_files = [f for f in os.listdir(IMAGE_ROOT) if os.path.splitext(f)[1] in SUPPORTED_IMAGE_EXT]

        for image_file in all_image_files:
            # 提取图片名（不含后缀）
            image_name = os.path.splitext(image_file)[0]
            # 拼接对应标签文件路径
            label_file = f"{image_name}.txt"
            label_path = os.path.join(LABEL_ROOT, label_file)

            # 检查标签文件是否存在
            if os.path.exists(label_path):
                matched_files.append({
                    "image": os.path.join(IMAGE_ROOT, image_file),
                    "label": label_path,
                    "name": image_name,
                    "image_ext": os.path.splitext(image_file)[1]
                })

        # 4. 检查有效数据量
        total_count = len(matched_files)
        if total_count == 0:
            raise ValueError("【错误】未找到任何图片-标签匹配的文件！请检查文件名是否一致")
        print(f"【信息】找到有效图片-标签对：{total_count} 个")

        # 5. 随机打乱并划分（固定种子保证可复现）
        random.seed(RANDOM_SEED)
        random.shuffle(matched_files)

        train_count = int(total_count * TRAIN_RATIO)
        train_files = matched_files[:train_count]
        val_files = matched_files[train_count:]

        print(f"【信息】划分结果：")
        print(f"  - 训练集：{len(train_files)} 个 ({TRAIN_RATIO * 100}%)")
        print(f"  - 测试集：{len(val_files)} 个 ({(1 - TRAIN_RATIO) * 100}%)")

        # 6. 复制文件到对应目录
        def copy_files(file_list, dst_image_dir, dst_label_dir, set_name):
            """批量复制文件"""
            copied_count = 0
            for file_info in file_list:
                try:
                    # 复制图片
                    dst_image_path = os.path.join(dst_image_dir, f"{file_info['name']}{file_info['image_ext']}")
                    shutil.copy(file_info["image"], dst_image_path)

                    # 复制标签
                    dst_label_path = os.path.join(dst_label_dir, f"{file_info['name']}.txt")
                    shutil.copy(file_info["label"], dst_label_path)

                    copied_count += 1
                except Exception as e:
                    print(f"【警告】复制文件失败 {file_info['name']}：{str(e)}")

            print(f"【信息】{set_name}集复制完成：{copied_count}/{len(file_list)} 个文件")

        # 复制训练集
        copy_files(train_files, train_image_dir, train_label_dir, "训练")
        # 复制测试集
        copy_files(val_files, val_image_dir, val_label_dir, "测试")

        # 7. 生成划分清单（可选，方便核对）
        list_file = os.path.join(OUTPUT_ROOT, "split_list.txt")
        with open(list_file, "w", encoding="utf-8") as f:
            f.write("===== 训练集文件 =====\n")
            for f_info in train_files:
                f.write(f"{f_info['name']}{f_info['image_ext']}\n")
            f.write("\n===== 测试集文件 =====\n")
            for f_info in val_files:
                f.write(f"{f_info['name']}{f_info['image_ext']}\n")

        print(f"\n【完成】所有操作执行完毕！")
        print(f"  - 划分后文件路径：{OUTPUT_ROOT}")
        print(f"  - 划分清单文件：{list_file}")

    except Exception as e:
        print(f"【错误】程序执行失败：{str(e)}")
        raise


# ===================== 主函数入口 =====================
if __name__ == "__main__":
    # 执行划分
    split_and_copy_dataset()