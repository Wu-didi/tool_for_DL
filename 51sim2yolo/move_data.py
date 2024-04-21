import os
import random
import shutil

# 指定图像和标签的文件夹路径
image_folder = r'E:\python_filev2\SyntheticDataset\images'
label_folder = r'E:\python_filev2\SyntheticDataset\labels'

# 指定验证集的保存路径
valid_image_folder = 'validate/images/'
valid_label_folder = 'validate/labels/'

# 确保验证集文件夹存在
if not os.path.exists(valid_image_folder):
    os.makedirs(valid_image_folder)
if not os.path.exists(valid_label_folder):
    os.makedirs(valid_label_folder)

# 获取图像和标签文件列表
image_files = os.listdir(image_folder)
label_files = os.listdir(label_folder)

# 计算要抽取的文件数量
num_files = len(image_files)
print(num_files)
num_validate = int(0.2 * num_files)  # 20%的文件作为验证集
print(num_validate)

# 随机抽取验证集文件
validate_files = random.sample(list(zip(image_files, label_files)), num_validate)

# 将验证集文件移动到验证集文件夹中
for image_file, label_file in validate_files:
    shutil.move(os.path.join(image_folder, image_file), os.path.join(valid_image_folder, image_file))
    shutil.move(os.path.join(label_folder, label_file), os.path.join(valid_label_folder, label_file))

print("Validation dataset created successfully!")
