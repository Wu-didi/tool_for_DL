import json
import os
import cv2
import tqdm

# 读取 JSON 文件
json_file_path = r'F:\wudi\dataset\traffic_sign\tt100k_2021\annotations_all.json'  # 替换为实际路径
with open(json_file_path, 'r') as f:
    data = json.load(f)

# 统计所有独特类别
unique_categories = set()
for img_info in tqdm.tqdm(data['imgs'].values()):
    for obj in img_info['objects']:
        unique_categories.add(obj['category'])

# 创建类别映射
category_mapping = {category: idx for idx, category in enumerate(sorted(unique_categories))}



# 创建输出目录
output_train_dir = 'output/yolo_labels/train'
output_test_dir = 'output/yolo_labels/test'
os.makedirs(output_train_dir, exist_ok=True)
os.makedirs(output_test_dir, exist_ok=True)

# 将类别映射写入 JSON 文件
mapping_file_path = 'output/category_mapping.json'
with open(mapping_file_path, 'w') as mapping_file:
    json.dump(category_mapping, mapping_file, indent=4)
# 处理每个图像
for img_id, img_info in tqdm.tqdm(data['imgs'].items()):
    img_path = img_info['path']
    objects = img_info['objects']
    
    # 读取图像以获取宽度和高度
    img = cv2.imread(img_path)
    img_height, img_width = img.shape[:2]

    # 根据路径决定输出目录
    if 'train' in img_path:
        output_dir = output_train_dir
    elif 'test' in img_path:
        output_dir = output_test_dir
    else:
        continue  # 如果路径中不包含train或test，则跳过

    # 创建 YOLO 格式标签文件
    yolo_label_file = os.path.join(output_dir, os.path.basename(img_path).replace('.jpg', '.txt'))
    
    with open(yolo_label_file, 'w') as yolo_file:
        for obj in objects:
            category = obj['category']
            if category in category_mapping:
                category_id = category_mapping[category]
                
                # 获取边界框
                xmin = obj['bbox']['xmin']
                ymin = obj['bbox']['ymin']
                xmax = obj['bbox']['xmax']
                ymax = obj['bbox']['ymax']
                
                # 计算中心坐标和宽高
                x_center = (xmin + xmax) / 2
                y_center = (ymin + ymax) / 2
                width = xmax - xmin
                height = ymax - ymin
                
                # 相对坐标
                x_center_rel = x_center / img_width
                y_center_rel = y_center / img_height
                width_rel = width / img_width
                height_rel = height / img_height
                
                # 写入 YOLO 格式
                yolo_file.write(f"{category_id} {x_center_rel} {y_center_rel} {width_rel} {height_rel}\n")

# 输出所有类别及其 ID
print("类别统计已写入文件：", mapping_file_path)
