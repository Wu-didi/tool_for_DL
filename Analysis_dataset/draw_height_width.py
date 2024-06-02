import os
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from collections import defaultdict

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
    
    return class_counts, class_heights, class_widths, class_aspect_ratios

def plot_class_counts(class_counts):
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=classes, y=counts, palette='viridis')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title('Object Count per Class')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_bbox_heatmap(class_widths, class_heights):
    for class_id in class_widths.keys():
        plt.figure(figsize=(10, 8))
        sns.kdeplot(x=class_widths[class_id], y=class_heights[class_id], cmap='viridis', fill=True, thresh=0.05)
        plt.xlabel('Width (pixels)')
        plt.ylabel('Height (pixels)')
        plt.title(f'Heatmap of Width vs Height for Class {class_id}')
        plt.tight_layout()
        plt.show()

def plot_bbox_distribution_for_dataset(data_dir):
    if not os.path.isdir(data_dir):
        raise NotADirectoryError(f"{data_dir} is not a valid directory.")
    
    label_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.txt')]
    
    if not label_files:
        raise FileNotFoundError(f"No label files found in directory {data_dir}.")
    
    class_counts, class_heights, class_widths, class_aspect_ratios = collect_class_statistics(label_files)
    plot_class_counts(class_counts)
    plot_bbox_heatmap(class_widths, class_heights)

data_dir = "/home/wudi/python_files/hefei_make_dataset/hefei_yolo_format_v2.0/train/labels"
plot_bbox_distribution_for_dataset(data_dir)
