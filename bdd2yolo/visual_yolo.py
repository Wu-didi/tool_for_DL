import cv2
import numpy as np

# 解析YOLO标注文件
def parse_yolo_annotation(annotation_file):
    with open(annotation_file, 'r') as file:
        lines = file.readlines()
    annotations = []
    for line in lines:
        parts = line.strip().split()
        class_id = int(parts[0])
        x, y, width, height = map(float, parts[1:])
        annotations.append((class_id, x, y, width, height))
    return annotations

# 可视化YOLO数据集
def visualize_yolo_dataset(image_path, annotation_path):
    image = cv2.imread(image_path)
    annotations = parse_yolo_annotation(annotation_path)
    
    for annotation in annotations:
        class_id, x, y, width, height = annotation
        img_h, img_w, _ = image.shape
        
        # 计算边界框的坐标
        x1 = int((x - width / 2) * img_w)
        y1 = int((y - height / 2) * img_h)
        x2 = int((x + width / 2) * img_w)
        y2 = int((y + height / 2) * img_h)
        
        # 在图像上绘制边界框
        color = (0, 255, 0)  # 绿色
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # 添加类别标签
        label = f"Class: {class_id}"
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # 显示图像
    cv2.imshow("YOLO Dataset", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 示例用法
image_path = r"E:\python_file\bdd2yolo\oneBDDdataset\images\0000f77c-6257be58.jpg"
annotation_path = r"E:\python_file\bdd2yolo\oneBDDdataset\label\0000f77c-6257be58.txt"
visualize_yolo_dataset(image_path, annotation_path)
