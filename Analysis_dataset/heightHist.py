import os
import matplotlib.pyplot as plt
from PIL import Image

def read_yolo_label_file(file_path, img_width, img_height):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        labels = []
        for line in lines:
            parts = line.strip().split(' ')
            bbox_height = float(parts[4]) * img_height
            bbox_width = float(parts[3]) * img_width
            label = {
                'class': parts[0],
                'x_center': float(parts[1]),
                'y_center': float(parts[2]),
                'width': float(parts[3]),
                'height': float(parts[4]),
                'pix_width': float(bbox_width),
                'pix_height': float(bbox_height)
            }
            # 计算bbox高度的像素点数
            
            labels.append(label)
        return labels
def get_image_size_from_label(label_file_path):
    # 获取图像文件路径
    img_file_path =  os.path.splitext(label_file_path)[0] + '.jpg'
    img_file_path = img_file_path.replace("labels","images")
    # 读取图像文件
    with Image.open(img_file_path) as img:
        # 获取图像尺寸
        img_width, img_height = img.size
    return img_width, img_height

def plot_bbox_height_distribution(yolo_label_files):
    heights = []
    for file in yolo_label_files:
        img_width, img_height = get_image_size_from_label(file)
        labels = read_yolo_label_file(file, img_width, img_height)
        for label in labels:
            bbox_height = label['pix_height']
            heights.append(bbox_height)
    # 绘制直方图
    plt.hist(heights, bins=50, density=True)
    plt.xlabel('bbox height (pixel count)')
    plt.ylabel('density')
    plt.title('bbox height distribution')
    plt.show()

def plot_bbox_aspect_ratio_distribution(yolo_label_files):
    aspect_ratios = []
    for file in yolo_label_files:
        img_width, img_height = get_image_size_from_label(file)
        labels = read_yolo_label_file(file, img_width, img_height)
        for label in labels:
            if label['pix_width'] == 0:
                continue
            aspect_ratio = label['pix_height']/label['pix_width']
            aspect_ratios.append(aspect_ratio)
    # 绘制直方图
    plt.hist(aspect_ratios, bins=50, density=True)
    plt.xlabel('bbox height (pixel count)')
    plt.ylabel('density')
    plt.title('bbox height distribution')
    plt.show()

def plot_bbox_distribution_for_dataset(data_dir):
    label_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.txt')]
    plot_bbox_height_distribution(label_files)
    plot_bbox_aspect_ratio_distribution(label_files)

data_dir = './mask2/labels/train'
plot_bbox_distribution_for_dataset(data_dir)
