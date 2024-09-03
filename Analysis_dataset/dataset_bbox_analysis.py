'''这段代码的主要功能是对YOLO格式标注的目标检测数据集进行统计分析，包括计算每个类别的目标框（bounding box）大小、长宽比的分布，
并生成相应的直方图和CSV文件。以下是代码的主要步骤和功能总结'''
import os
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from collections import defaultdict
import numpy as np

# 类别列表
class_id_index =  ['traffic-signal-system_good', 
'traffic-signal-system_bad', 
'traffic-guidance-system_good',
 'traffic-guidance-system_bad',
  'restricted-elevated_good', 
  'restricted-elevated_bad', 
  'cabinet_good', 
  'cabinet_bad', 
  'backpack-box_good', 
  'backpack-box_bad', 
  'off-site', 
  'Gun-type-Camera', 
  'Dome-Camera', 
  'Flashlight', 
  'b-Flashlight']

def read_yolo_label_file(file_path, img_width, img_height):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        labels = []
        for line in lines:
            parts = line.strip().split(' ')
            class_id = parts[0]
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            
            bbox_width = width * img_width
            bbox_height = height * img_height

            if class_id in ['14']:  # 排除“b-Flashlight”类别
                continue

            # 合并_good和_bad类别
            if class_id.endswith('_good') or class_id.endswith('_bad'):
                class_id = class_id[:-5]

            label = {
                'class': class_id,
                'x_center': x_center,
                'y_center': y_center,
                'width': width,
                'height': height,
                'pix_width': bbox_width,
                'pix_height': bbox_height
            }
            
            labels.append(label)
        return labels

def get_image_size_from_label(label_file_path):
    img_file_path = os.path.splitext(label_file_path)[0] + '.jpg'
    img_file_path = img_file_path.replace("labels", "images")
    
    if not os.path.exists(img_file_path):
        raise FileNotFoundError(f"Image file {img_file_path} not found.")
    
    with Image.open(img_file_path) as img:
        img_width, img_height = img.size
    
    return img_width, img_height

def remove_outliers(data):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return [x for x in data if lower_bound <= x <= upper_bound]

def collect_class_statistics(yolo_label_files):
    class_counts = defaultdict(int)
    class_heights = defaultdict(list)
    class_widths = defaultdict(list)
    class_aspect_ratios = defaultdict(list)
    
    for file in yolo_label_files:
        img_width, img_height = get_image_size_from_label(file)
        labels = read_yolo_label_file(file, img_width, img_height)
        for label in labels:
            class_id = label['class']
            class_counts[class_id] += 1
            class_heights[class_id].append(label['pix_height'])
            class_widths[class_id].append(label['pix_width'])
            if label['pix_width'] != 0:
                class_aspect_ratios[class_id].append(label['pix_height'] / label['pix_width'])
    
    # 去除宽度和高度中的异常值
    for key in class_widths:
        class_widths[key] = remove_outliers(class_widths[key])
    for key in class_heights:
        class_heights[key] = remove_outliers(class_heights[key])
    
    return class_counts, class_heights, class_widths, class_aspect_ratios

def save_aspect_ratios_to_csv(class_aspect_ratios, output_dir):
    csv_file_path = os.path.join(output_dir, 'aspect_ratios.csv')
    
    # 找到长宽比最多的类别，确定列数
    max_len = max(len(ratios) for ratios in class_aspect_ratios.values())

    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([class_id_index[int(class_id)] for class_id in class_aspect_ratios.keys()])
        
        for i in range(max_len):
            row = []
            for class_id in class_aspect_ratios.keys():
                if i < len(class_aspect_ratios[class_id]):
                    row.append(class_aspect_ratios[class_id][i])
                else:
                    row.append('')
            writer.writerow(row)

def plot_bbox_size_distribution(class_widths, class_heights, output_dir):
    all_widths = []
    all_heights = []
    
    for widths in class_widths.values():
        all_widths.extend(widths)
    
    for heights in class_heights.values():
        all_heights.extend(heights)
    
    plt.figure(figsize=(14, 6))
    
    # 绘制宽度分布直方图
    plt.subplot(1, 2, 1)
    sns.histplot(all_widths, bins=30, kde=True, color='skyblue')
    plt.xlabel('Width (pixels)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Bounding Box Widths')
    
    # 绘制高度分布直方图
    plt.subplot(1, 2, 2)
    sns.histplot(all_heights, bins=30, kde=True, color='salmon')
    plt.xlabel('Height (pixels)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Bounding Box Heights')
    
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, 'bbox_size_distribution.png'), dpi=300)
    plt.close()

def plot_aspect_ratio_distribution(class_aspect_ratios, output_dir):
    num_classes = len(class_aspect_ratios)
    cols = 3
    rows = (num_classes + cols - 1) // cols
    
    plt.figure(figsize=(5 * cols, 4 * rows))
    
    for idx, (class_id, ratios) in enumerate(class_aspect_ratios.items(), 1):
        plt.subplot(rows, cols, idx)
        sns.histplot(ratios, bins=30, kde=True, color='purple')
        plt.xlabel('Aspect Ratio (Height/Width)')
        plt.ylabel('Frequency')
        plt.title(f'Aspect Ratio Distribution for Class {class_id_index[int(class_id)]}')
    
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, 'aspect_ratio_distribution.png'), dpi=300)
    plt.close()

def plot_bbox_distribution_for_dataset(data_dir, output_dir):
    if not os.path.isdir(data_dir):
        raise NotADirectoryError(f"{data_dir} is not a valid directory.")
    
    label_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.txt')]
    
    if not label_files:
        raise FileNotFoundError(f"No label files found in directory {data_dir}.")
    
    class_counts, class_heights, class_widths, class_aspect_ratios = collect_class_statistics(label_files)
    
    # 保存长宽比到CSV文件
    print(class_aspect_ratios.keys())
    save_aspect_ratios_to_csv(class_aspect_ratios, output_dir)
    
    plot_bbox_size_distribution(class_widths, class_heights, output_dir)
    plot_aspect_ratio_distribution(class_aspect_ratios, output_dir)

# 目录路径设置
data_dir = "/data/hefei-dataset/hefei_yolo_format_v2.4/train/labels"
output_dir = "./"

# 生成并保存直方图和CSV文件
plot_bbox_distribution_for_dataset(data_dir, output_dir)
