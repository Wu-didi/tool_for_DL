'''每个类别的目标框的长、宽和长宽比统计信息就分别保存在不同的CSV文件中,并绘制每个类别的直方图'''

import os
import csv
import matplotlib.pyplot as plt

label_dir = './mask2/labels/train'
img_dir = './mask2/images/train'

# 字典用于存储每个类别的长宽和长宽比信息
class_stats = {}

# 遍历标签文件夹中所有标签文件
for label_file in os.listdir(label_dir):
    if not label_file.endswith(".txt"):
        continue

    # 读取标签文件中的目标框信息
    with open(os.path.join(label_dir, label_file), "r") as f:
        lines = f.readlines()
    
    # 遍历目标框信息中所有目标框
    for line in lines:
        # 将目标框信息解析为坐标和类别信息
        parts = line.strip().split(" ")
        class_id = int(parts[0])
        x, y, w, h = map(float, parts[1:])
        
        # 跳过高度为0的框，避免除零错误每个类别的目标框的长、宽和长宽比统计信息就分别保存在不同的CSV文件中
        # 计算目标框的长宽比
        aspect_ratio = w / h
        
        # 如果类别ID不在字典中，则初始化一个新的列表
        if class_id not in class_stats:
            class_stats[class_id] = []

        # 将长、宽和长宽比添加到相应类别的列表中
        class_stats[class_id].append((w, h, aspect_ratio))

# 为每个类别生成一个CSV文件，并将统计信息写入其中
for class_id, stats in class_stats.items():
    csv_filename = f'class_{class_id}_stats.csv'
    with open(csv_filename, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Width', 'Height', 'Aspect Ratio'])
        writer.writerows(stats)

    print(f'Saved statistics for class {class_id} to {csv_filename}')

    # 提取当前类别的长宽比信息
    aspect_ratios = [stat[2] for stat in stats]

    # 绘制长宽比直方图
    plt.figure()
    plt.hist(aspect_ratios, bins=100, range=(0, 5), density=True)
    plt.xlabel("Aspect Ratio")
    plt.ylabel("Probability")
    plt.title(f'Aspect Ratio Distribution for Class {class_id}')
    
    # 保存直方图图像
    hist_filename = f'class_{class_id}_aspect_ratio_histogram.png'
    plt.savefig(hist_filename)
    plt.close()

    print(f'Saved aspect ratio histogram for class {class_id} to {hist_filename}')
