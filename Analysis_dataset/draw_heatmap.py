import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 数据集路径
dataset_path = r"./mask2"
image_folder = os.path.join(dataset_path, "images/train")
label_folder = os.path.join(dataset_path, "labels/train")

# 设置热力图大小和精度
heatmap_size = (640, 640)
heatmap_precision = 4

# 初始化热力图
heatmap = np.zeros(heatmap_size, dtype=np.float32)
i=1
# 遍历标签文件夹，计算每个标注框的位置
for label_file in os.listdir(label_folder):

    print("label_file",label_file)
    print(i)
    i+=1
    label_path  = os.path.join(label_folder,label_file)
    # 获取图片文件名
    image_file = label_file.replace(".txt", ".jpg")
    image_path = os.path.join(image_folder,image_file)
    print(image_path)
    # 读取图片，获取宽高
    image = cv2.imread(image_path)
    image_height, image_width, _ = image.shape
    # 读取标签文件，获取标注框位置
    with open(os.path.join(label_folder, label_file), 'r') as f:
        for line in f.readlines():
            line = line.strip().split()
            x_center = float(line[1]) * image_width
            y_center = float(line[2]) * image_height
            width = float(line[3]) * image_width
            height = float(line[4]) * image_height
            # 将标注框位置映射到热力图上
            x1 = int(max(0, (x_center - width / 2) * heatmap_size[0] / image_width))
            y1 = int(max(0, (y_center - height / 2) * heatmap_size[1] / image_height))
            x2 = int(min(heatmap_size[0] - 1, (x_center + width / 2) * heatmap_size[0] / image_width))
            y2 = int(min(heatmap_size[1] - 1, (y_center + height / 2) * heatmap_size[1] / image_height))
            # 更新热力图
            heatmap[y1:y2+1, x1:x2+1] += 1

# 将热力图标准化到[0, 1]
heatmap /= np.max(heatmap)

# 将热力图转换成RGB格式
heatmap_rgb = cv2.applyColorMap(np.uint8(heatmap*255), cv2.COLORMAP_JET)

# 将热力图和原图叠加在一起
image = cv2.imread(os.path.join(image_folder, os.listdir(image_folder)[0]))
image = cv2.resize(image,(640,640))

#设置纯黑色的背景进行叠加
# image = np.zeros((heatmap_size[0], heatmap_size[1], 3), dtype=np.uint8)
# image[:, :] = (0, 0, 0)


print(heatmap_rgb.shape)
print(image.shape)
heatmap_with_image = cv2.addWeighted(heatmap_rgb, 0.5, image, 0.5, 0)

# 显示热力图和原图叠加在一起的结果
plt.imshow(cv2.cvtColor(heatmap_with_image, cv2.COLOR_BGR2RGB))
plt.show()
