import json
import cv2
import numpy as np

# 读取JSON文件
with open(r'E:\python_filev2\yolov5-6.0\datasets\train\BDD_style_labels\000000.json') as json_file:
    data = json.load(json_file)

# 读取对应的图像
image_path = r'E:\python_filev2\yolov5-6.0\datasets\train\images\000000.png'
image = cv2.imread(image_path)
print(image.shape)
# 输出宽度和高度
print('Image width:', image.shape[1])
print('Image height:', image.shape[0])

# 绘制2D框和标签
for bbox in data['bboxes']:
    # 提取框的位置和大小
    x1, y1, x2, y2 = bbox['bbox']
    
    # 提取框的类型
    label_type = bbox['type']
    
    # 绘制矩形框
    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    
    # 添加标签
    cv2.putText(image, f'Type: {label_type}', (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

# 显示图像
cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
