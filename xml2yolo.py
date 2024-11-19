import os
import xml.etree.ElementTree as ET
'''合肥项目代码'''
# 定义类别
classes = [
    "veh_normal_signal",
    "per_normal_signal",
    "veh_multi_signal",
    "per_multi_signal",
    "veh_abnormal_signal",
    "per_abnormal_signal",
    "red",
    "yellow",
    "green"
]

# 读取 XML 文件并转换为 YOLO 格式
def xml_to_yolo(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # 从 XML 获取图像宽度和高度
    size = root.find('size')
    img_width = int(size.find('width').text)
    img_height = int(size.find('height').text)

    yolo_labels = []
    
    for obj in root.findall('object'):
        class_name = obj.find('name').text
        if class_name not in classes:
            continue
        class_index = classes.index(class_name)

        # 获取边界框坐标
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)

        # 计算 YOLO 格式的坐标
        x_center = (xmin + xmax) / 2 / img_width
        y_center = (ymin + ymax) / 2 / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height

        # 添加到标签列表
        yolo_labels.append(f"{class_index} {x_center} {y_center} {width} {height}")

    return img_width, img_height, yolo_labels

# 处理文件夹中的所有 XML 文件
def process_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".xml"):
            xml_file = os.path.join(input_folder, filename)

            # 从 XML 文件获取图像宽度和高度以及 YOLO 格式标签
            img_width, img_height, yolo_format_labels = xml_to_yolo(xml_file)
            yolo_label_file = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.txt")

            # 保存 YOLO 格式标签
            with open(yolo_label_file, 'w') as f:
                for label in yolo_format_labels:
                    f.write(label + '\n')

# 指定包含 XML 文件的输入文件夹路径和输出文件夹路径
input_folder_path = r"E:\dataset\hefei_stage2\part_labeled\v3\Annotations"
output_folder_path = r"E:\dataset\hefei_stage2\part_labeled\v3\yolo_label"
process_folder(input_folder_path, output_folder_path)
