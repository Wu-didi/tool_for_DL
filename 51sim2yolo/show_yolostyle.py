import os
import cv2

# 指定图像和标签的路径
image_folder = r'E:\BaiduNetdiskDownload\Dataset\10vehicle-image\train\10-vehicle\image'
label_folder = r'./labels/'

# 选择一个图像
image_filename = '000000.png'

# 构建图像和标签文件的完整路径
image_path = os.path.join(image_folder, image_filename)
print(image_path)
label_path = os.path.join(label_folder, os.path.splitext(image_filename)[0] + '.txt')

# 读取图像
image = cv2.imread(image_path)

# 获取图像的宽度和高度
image_height, image_width, _ = image.shape

# 读取YOLO格式标签文件
with open(label_path, 'r') as f:
    lines = f.readlines()

# 绘制边界框
for line in lines:
    label = line.split()
    label_type = int(label[0])
    x_center, y_center, bbox_width, bbox_height = map(float, label[1:])
    
    # 计算边界框的左上角和右下角坐标
    x1 = int((x_center - bbox_width / 2) * image_width)
    y1 = int((y_center - bbox_height / 2) * image_height)
    x2 = int((x_center + bbox_width / 2) * image_width)
    y2 = int((y_center + bbox_height / 2) * image_height)
    
    # 绘制矩形框
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

# 显示图像
cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
