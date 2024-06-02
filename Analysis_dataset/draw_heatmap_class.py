import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 数据集路径
dataset_path = r"/home/wudi/python_files/hefei_make_dataset/hefei_yolo_format_v2.0"
image_folder = os.path.join(dataset_path, "val/images")
label_folder = os.path.join(dataset_path, "val/labels")

# 设置热力图大小和精度
heatmap_size = (640, 640)

# 初始化类别热力图字典
class_heatmaps = {}

# 遍历标签文件夹，计算每个标注框的位置
for label_file in os.listdir(label_folder):
    label_path = os.path.join(label_folder, label_file)
    # 获取图片文件名
    image_file = label_file.replace(".txt", ".jpg")
    image_path = os.path.join(image_folder, image_file)

    # 读取图片，获取宽高
    image = cv2.imread(image_path)
    if image is None:
        continue
    image_height, image_width, _ = image.shape

    # 读取标签文件，获取标注框位置
    with open(label_path, 'r') as f:
        for line in f.readlines():
            line = line.strip().split()
            class_id = int(line[0])
            x_center = float(line[1]) * image_width
            y_center = float(line[2]) * image_height
            width = float(line[3]) * image_width
            height = float(line[4]) * image_height

            # 初始化类别热力图
            if class_id not in class_heatmaps:
                class_heatmaps[class_id] = np.zeros(heatmap_size, dtype=np.float32)

            # 将标注框位置映射到热力图上
            x1 = int(max(0, (x_center - width / 2) * heatmap_size[0] / image_width))
            y1 = int(max(0, (y_center - height / 2) * heatmap_size[1] / image_height))
            x2 = int(min(heatmap_size[0] - 1, (x_center + width / 2) * heatmap_size[0] / image_width))
            y2 = int(min(heatmap_size[1] - 1, (y_center + height / 2) * heatmap_size[1] / image_height))

            # 更新类别热力图
            class_heatmaps[class_id][y1:y2+1, x1:x2+1] += 1

# 为每个类别生成并保存热力图
for class_id, heatmap in class_heatmaps.items():
    # 将热力图标准化到[0, 1]
    heatmap /= np.max(heatmap)

    # 将热力图转换成RGB格式
    heatmap_rgb = cv2.applyColorMap(np.uint8(heatmap * 255), cv2.COLORMAP_JET)

    # 设置纯黑色的背景进行叠加
    image = np.zeros((heatmap_size[0], heatmap_size[1], 3), dtype=np.uint8)
    image[:, :] = (0, 0, 0)

    # 叠加热力图和背景
    heatmap_with_image = cv2.addWeighted(heatmap_rgb, 0.5, image, 0.5, 0)

    # 显示热力图和原图叠加在一起的结果
    plt.imshow(cv2.cvtColor(heatmap_with_image, cv2.COLOR_BGR2RGB))
    plt.title(f'Class {class_id} Heatmap')
    plt.show()

    # 保存热力图
    heatmap_filename = f"class_{class_id}_heatmap.jpg"
    cv2.imwrite(heatmap_filename, heatmap_with_image)
    print(f"Saved heatmap for class {class_id} to {heatmap_filename}")

'''
初始化类别热力图字典：用一个字典 class_heatmaps 来存储每个类别的热力图。

遍历标签文件：读取标签文件并解析每行数据，获取标注框信息。

根据类别ID初始化热力图：如果 class_id 还没有对应的热力图，则初始化一个新的热力图数组。

计算并更新热力图：根据标注框位置更新对应类别的热力图。

生成并保存热力图：遍历 class_heatmaps 字典，为每个类别生成热力图，将其标准化并转换成RGB格式，然后保存到文件中。

这样，每个类别的热力图都会分别生成并保存为图片文件。
'''