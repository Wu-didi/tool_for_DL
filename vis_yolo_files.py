import os
import cv2
from tqdm import tqdm
import numpy as np
import random

# 标签文件和图片文件的路径
label_folder = r'E:\dataset\hefei_stage2\part_labeled\data\labels'
image_folder = r'E:\dataset\hefei_stage2\part_labeled\data\images'
output_folder = r'E:\dataset\hefei_stage2\part_labeled\data\outputs'

# 如果输出文件夹不存在，创建它
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 定义类别数量
num_classes = 9  # 根据需要修改类别数量

# 随机生成颜色（BGR格式）
def generate_random_colors(num_classes):
    return [tuple(random.randint(0, 255) for _ in range(3)) for _ in range(num_classes)]

colors = generate_random_colors(num_classes)

# 获取图片文件列表
image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]

# 使用 tqdm 显示进度条
for filename in tqdm(image_files, desc="Processing images"):
    try:
        image_path = os.path.join(image_folder, filename)
        label_path = os.path.join(label_folder, filename.replace('.jpg', '.txt'))

        # 读取图片
        image = cv2.imread(image_path)
        height, width, _ = image.shape

        # 检查标签文件是否存在
        if not os.path.exists(label_path):
            print(f"Label file not found for {filename}, skipping...")
            continue

        # 读取标签文件
        with open(label_path, 'r') as f:
            lines = f.readlines()

        # 解析标签并绘制矩形框和类别 ID
        for line in lines:
            class_idx, x_center, y_center, w, h = map(float, line.strip().split())

            # 将归一化坐标转换为实际像素坐标
            x_center *= width
            y_center *= height
            w *= width
            h *= height

            # 计算矩形框的左上角和右下角坐标
            x1 = int(x_center - w / 2)
            y1 = int(y_center - h / 2)
            x2 = int(x_center + w / 2)
            y2 = int(y_center + h / 2)

            # 获取当前类别的颜色
            color = colors[int(class_idx)]

            # 在图片上绘制矩形框
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            # 在矩形框的左上角标注类别 ID
            label = f'ID: {int(class_idx)}'
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 保存带有矩形框的图片
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, image)

    except Exception as e:
        print(f"Error processing {filename}: {e}, skipping...")

print("处理完成，标注图片已保存到：", output_folder)
