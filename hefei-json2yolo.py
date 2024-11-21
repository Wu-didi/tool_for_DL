import os
import json
from PIL import Image  # 用于获取图片的宽高

# 标签映射到类别 ID
# label_map = {
#     "veh_normal_signal": 0,
#     "veh_multi_signal": 1,
#     "green": 2,
#     "red": 3,
#     "yellow": 4,
# }

label_map = {
    "veh_normal_signal":0,
    "per_normal_signal":1,
    "veh_multi_signal":2,
    "per_multi_signal":3,
    "veh_abnormal_signal":4,
    "per_abnormal_signal":5,
    "red":6,
    "yellow":7,
    "green":8,
}

# 输入和输出路径
json_folder = r"D:\东南\项目\合肥-基础设施检测项目\stage2_labeled\labels"  # JSON 文件夹路径
image_folder = r"D:\东南\项目\合肥-基础设施检测项目\stage2_labeled\images"  # 图片文件夹路径
output_folder = r"D:\东南\项目\合肥-基础设施检测项目\stage2_labeled\yolo_labels"  # 输出标签文件夹路径（如 train/labels 或 val/labels）

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 遍历 JSON 文件夹
for json_file in os.listdir(json_folder):
    if not json_file.endswith(".json"):
        continue

    json_path = os.path.join(json_folder, json_file)
    
    # 对应的图片文件
    image_name = os.path.splitext(json_file)[0] + ".jpg"
    image_path = os.path.join(image_folder, image_name)

    # 检查图片是否存在
    if not os.path.exists(image_path):
        print(f"图片 {image_name} 不存在，跳过该文件")
        continue

    # 获取图片的宽和高
    with Image.open(image_path) as img:
        image_width, image_height = img.size

    # 读取 JSON 文件
    with open(json_path, "r") as file:
        data = json.load(file)

    yolo_annotations = []

    for shape in data["shapes"]:
        label = shape["label"]
        points = shape["points"]
        if label not in label_map:
            continue

        # 计算矩形框中心点、宽度和高度
        x_min, y_min = points[0]
        x_max, y_max = points[1]

        x_center = ((x_min + x_max) / 2) / image_width
        y_center = ((y_min + y_max) / 2) / image_height
        width = (x_max - x_min) / image_width
        height = (y_max - y_min) / image_height

        # 转换为 YOLO 格式
        class_id = label_map[label]
        yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    # 保存为 .txt 文件
    output_file = os.path.join(output_folder, os.path.splitext(json_file)[0] + ".txt")
    with open(output_file, "w") as f:
        f.write("\n".join(yolo_annotations))

    print(f"已生成 {output_file}")

print("所有文件处理完成！")
