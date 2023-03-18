# import os
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# data_dir = './data'  # 数据集所在目录
# label_dir = './mask2/labels/train'
# img_dir = './mask2/images/train'
# aspect_ratios = []  # 存储长宽比的列表

# # 读取标签文件并统计长宽比
# for label_file in os.listdir(label_dir):
#     with open(os.path.join(label_dir, label_file), 'r') as f:
#         lines = f.readlines()
#         for line in lines:
#             line = line.strip().split()
#             x, y, w, h = map(float, line[1:])
#             if h == 0:
#                 continue
#             aspect_ratio = w / h
#             aspect_ratios.append(aspect_ratio)

# # 绘制直方图
# plt.hist(aspect_ratios, bins=100, density=True)
# plt.xlabel('Aspect ratio')
# plt.ylabel('Density')
# plt.title('Aspect ratio distribution')
# plt.show()
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 数据集文件夹情况为图片一个文件夹images，标签文件一个文件夹labels
label_dir = './mask2/labels/train'
img_dir = './mask2/images/train'

aspect_ratios = []

# 遍历标签文件夹中所有标签文件
for label_file in os.listdir(label_dir):
    if not label_file.endswith(".txt"):
        continue

    # 读取标签文件中的目标框信息
    with open(os.path.join(label_dir, label_file), "r") as f:
        lines = f.readlines()
    
    # 遍历目标框信息中所有目标框
    for line in lines:
        # 将目标框信息解析为坐标和类别信息
        parts = line.strip().split(" ")
        class_id = int(parts[0])
        x, y, w, h = map(float, parts[1:])
        
        # 计算目标框的长宽比
        if h == 0:
            continue
        aspect_ratio = w / h
        aspect_ratios.append(aspect_ratio)

# 绘制长宽比直方图
plt.hist(aspect_ratios, bins=100, range=(0, 5), density=True)
plt.xlabel("Aspect Ratio")
plt.ylabel("Probability")
plt.show()
