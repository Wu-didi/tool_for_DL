import os
import json
import tqdm
# 自定义NaN的解析函数
def custom_parse_float(value):
    if value.lower() == 'nan':
        return float('nan')
    return float(value)

# # 读取包含NaN的JSON数据
# with open('your_file.json', 'r') as f:
#     data = json.load(f, parse_float=custom_parse_float)

# # 打印数据
# print(data)
def convert_to_yolo_format(bbox, image_width, image_height):
    # 提取框的位置和大小
    x1, y1, x2, y2 = bbox['bbox']
    
    # 计算中心点坐标和宽度高度
    x_center = (x1 + x2) / (2 * image_width)
    y_center = (y1 + y2) / (2 * image_height)
    bbox_width = (x2 - x1) / image_width
    bbox_height = (y2 - y1) / image_height
    
    # 返回YOLO格式的边界框坐标
    return [x_center, y_center, bbox_width, bbox_height]

# 设置图像的宽度和高度
image_width = 1920  # 替换为你的图像宽度
image_height = 1080  # 替换为你的图像高度
# 定义类别字典，用于统计每个类别出现的次数
class_counts = {}
after_class_counts = {}


#################################需要修改下面的路径###############################################################
# 指定image_label文件夹的路径
label_folder = r'E:\BaiduNetdiskDownload\Dataset\10vehicle-image\train\10-vehicle\image_label'

# 指定保存YOLO格式标签的文件夹路径
output_folder = './labels/'

# 做一个类别的映射
label_dict = {   '6' : '0',
    '8' : '1',
    '13' : '2',
    '15' : '3',
    '18' : '4',
    '19' : '5',
    '21' : '6',
    '26' : '7',
    '27' : '8',}

# 创建保存YOLO格式标签的文件夹
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历image_label文件夹下的所有JSON文件
for filename in tqdm.tqdm(os.listdir(label_folder)):
    # print(filename)
    if filename.endswith('.json'):
        # 构建完整的文件路径
        file_path = os.path.join(label_folder, filename)
        
        # 读取JSON文件
        with open(file_path) as json_file:
            # data = json.load(json_file, parse_float=custom_parse_float)
            data = json_file.read()
            # print(data)
            # print(type(data))
            # 替换nan为0
            data = data.replace('nan', '0')
            # 替换-inf为0
            data = data.replace('-inf', '0')
            
            # str 转 dict
            
            data = json.loads(data)
        
        # 创建用于保存YOLO格式标签的列表
        yolo_labels = []
        
        # 转换每个边界框为YOLO格式
        for bbox in data['bboxes']:
            yolo_bbox = convert_to_yolo_format(bbox, image_width, image_height)
            label_type = bbox['type']
            # 统计类别
                    # 统计类别出现次数
            if label_type in class_counts:
                class_counts[label_type] += 1
            else:
                class_counts[label_type] = 1
            
            # 将类别映射为整数
            label_type = label_dict[str(label_type)]
            # 将标签组织成YOLO格式并添加到列表
            print(int(label_type))
            yolo_labels.append([int(label_type)] + yolo_bbox)
            
            if label_type in after_class_counts:
                after_class_counts[label_type] += 1
            else:
                after_class_counts[label_type] = 1
        
        # 构建保存YOLO格式标签的文件路径
        yolo_filename = os.path.splitext(filename)[0] + '.txt'
        yolo_filepath = os.path.join(output_folder, yolo_filename)
        
        # 保存到文件
        with open(yolo_filepath, 'w') as f:
            for label in yolo_labels:
                f.write(' '.join([str(x) for x in label]) + '\n')

print("Conversion complete!")
# 打印类别统计结果
print("Class Counts:")
for class_id, count in class_counts.items():
    print(f"Class {class_id}: {count} instances")

print("After Class Counts:")
for class_id, count in after_class_counts.items():
    print(f"Class {class_id}: {count} instances")