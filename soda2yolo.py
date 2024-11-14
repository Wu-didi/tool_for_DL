import json
import os

# 定义类别映射函数
def get_category_map(categories):
    print(categories)
    return {category["id"]: category["name"] for category in categories}

# YOLO 格式转换函数
def convert_to_yolo(annotation, image_width, image_height):
    category_id = annotation["category_id"]
    bbox = annotation["bbox"]
    
    # 计算中心坐标和相对尺寸
    x_min, y_min, width, height = bbox
    center_x = (x_min + width / 2) / image_width
    center_y = (y_min + height / 2) / image_height
    norm_width = width / image_width
    norm_height = height / image_height
    
    # 类别ID（YOLO格式从0开始，所以需要减去1）
    class_id = category_id - 1
    
    return f"{class_id} {center_x} {center_y} {norm_width} {norm_height}"

# 读取JSON文件并生成YOLO标签
def generate_yolo_labels(json_file, output_folder, image_folder):
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # 获取类别映射
    category_map = get_category_map(data["categories"])

    # 统计每个类别的数量
    category_counts = {category["id"]: 0 for category in data["categories"]}

    # 遍历每张图片，处理注释
    for image in data["images"]:
        image_id = image["id"]
        image_width = image["width"]
        image_height = image["height"]
        image_filename = image["file_name"]

        # 找到该图片对应的所有注释
        annotations_for_image = [
            annotation for annotation in data["annotations"] if annotation["image_id"] == image_id
        ]
        
        # 创建该图片的YOLO标签文件
        yolo_labels = []
        for annotation in annotations_for_image:
            yolo_label = convert_to_yolo(annotation, image_width, image_height)
            yolo_labels.append(yolo_label)

            # 更新类别计数
            category_counts[annotation["category_id"]] += 1
        
        # 写入YOLO标签文件
        label_filename = os.path.splitext(image_filename)[0] + '.txt'
        label_filepath = os.path.join(output_folder, label_filename)
        
        with open(label_filepath, 'w') as label_file:
            label_file.write("\n".join(yolo_labels))
    
    return category_counts

# 处理数据集中的所有注释文件
def process_dataset(dataset_folder):
    annotations_folder = os.path.join(dataset_folder, "annotations")
    
    # 统计每个数据集的类别数量
    all_category_counts = {category["id"]: 0 for category in json.load(open(os.path.join(annotations_folder, 'instance_train.json')))["categories"]}
    
    # 处理训练集、验证集和测试集的注释文件
    for split in ['train', 'val']:
        annotation_file = os.path.join(annotations_folder, f'instance_{split}.json')
        output_folder = os.path.join(dataset_folder, split, 'labels')  # 为每个数据集创建 'labels' 子文件夹
        
        # 创建输出文件夹（如果不存在）
        os.makedirs(output_folder, exist_ok=True)
        
        # 生成YOLO格式的标签并统计类别数量
        category_counts = generate_yolo_labels(annotation_file, output_folder, output_folder)
        
        # 将每个数据集的统计结果合并到总的统计结果中
        for category_id, count in category_counts.items():
            all_category_counts[category_id] += count
    
    # 输出每个类别的数量
    return all_category_counts

# 运行转换
dataset_folder = './'  # 修改为实际的dataset路径
category_counts = process_dataset(dataset_folder)

# 输出统计结果
for category_id, count in category_counts.items():
    print(f"Category ID {category_id}: {count} objects")
