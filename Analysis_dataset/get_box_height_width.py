import os
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import numpy as np

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

            label = {
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

def collect_statistics(yolo_label_files):
    all_widths = []
    all_heights = []
    all_aspect_ratios = []
    
    for file in yolo_label_files:
        img_width, img_height = get_image_size_from_label(file)
        labels = read_yolo_label_file(file, img_width, img_height)
        for label in labels:
            all_widths.append(label['pix_width'])
            all_heights.append(label['pix_height'])
            if label['pix_width'] != 0:
                all_aspect_ratios.append(label['pix_height'] / label['pix_width'])
    
    # 去除宽度和高度中的异常值
    all_widths = remove_outliers(all_widths)
    all_heights = remove_outliers(all_heights)
    
    return all_widths, all_heights, all_aspect_ratios

def save_bbox_dimensions_to_csv(all_widths, all_heights, output_dir):
    csv_file_path = os.path.join(output_dir, 'bbox_dimensions.csv')
    
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Width (pixels)', 'Height (pixels)'])
        for width, height in zip(all_widths, all_heights):
            writer.writerow([width, height])

def plot_combined_statistics(all_widths, all_heights, all_aspect_ratios, output_dir):
    plt.figure(figsize=(14, 6))
    
    # 绘制宽度分布直方图
    plt.subplot(1, 2, 1)
    sns.histplot(all_widths, bins=30, kde=True, color='skyblue')
    plt.xlabel('Width (pixels)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Bounding Box Widths (All Classes Combined)')
    
    # 绘制高度分布直方图
    plt.subplot(1, 2, 2)
    sns.histplot(all_heights, bins=30, kde=True, color='salmon')
    plt.xlabel('Height (pixels)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Bounding Box Heights (All Classes Combined)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'bbox_size_distribution_combined.png'), dpi=300)
    plt.close()

    # 绘制长宽比分布直方图
    plt.figure(figsize=(7, 6))
    sns.histplot(all_aspect_ratios, bins=30, kde=True, color='purple')
    plt.xlabel('Aspect Ratio (Height/Width)')
    plt.ylabel('Frequency')
    plt.title('Aspect Ratio Distribution (All Classes Combined)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'aspect_ratio_distribution_combined.png'), dpi=300)
    plt.close()

def plot_bbox_distribution_for_dataset(data_dir, output_dir):
    if not os.path.isdir(data_dir):
        raise NotADirectoryError(f"{data_dir} is not a valid directory.")
    
    label_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.txt')]
    
    if not label_files:
        raise FileNotFoundError(f"No label files found in directory {data_dir}.")
    
    all_widths, all_heights, all_aspect_ratios = collect_statistics(label_files)
    
    # 保存长和宽到CSV文件
    save_bbox_dimensions_to_csv(all_widths, all_heights, output_dir)
    
    plot_combined_statistics(all_widths, all_heights, all_aspect_ratios, output_dir)

# 目录路径设置
data_dir = "/data/hefei-dataset/hefei_yolo_format_v2.4/train/labels"
output_dir = "./"

# 生成并保存直方图和CSV文件
plot_bbox_distribution_for_dataset(data_dir, output_dir)
