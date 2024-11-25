"根据数据集中，类别数量，设置优先级，数量少的先划分，保证按照类别分层划分数据集"

import os
import shutil
from collections import Counter, defaultdict
from tqdm import tqdm
import random

# 统计各类别的数量
def count_labels(labels_dir):
    label_counts = Counter()
    image_to_labels = defaultdict(list)

    for filename in os.listdir(labels_dir):
        if filename.endswith('.txt'):
            label_file = os.path.join(labels_dir, filename)
            with open(label_file, 'r') as f:
                labels = [int(line.split()[0]) for line in f]
                label_counts.update(labels)  # 更新类别计数
                image_to_labels[filename.replace('.txt', '.jpg')] = labels  # 保存图片与类别对应关系

    return label_counts, image_to_labels

# 自定义划分函数
def custom_split(data_list, train_ratio, val_ratio):
    random.shuffle(data_list)
    total = len(data_list)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_data = data_list[:train_end]
    val_data = data_list[train_end:val_end]
    test_data = data_list[val_end:]

    return train_data, val_data, test_data

# 按类别依次划分数据集
def stratified_split_by_category(label_counts, image_to_labels, train_ratio=0.7, val_ratio=0.2):
    sorted_labels = sorted(label_counts.items(), key=lambda x: x[1])  # 按类别数量排序
    mask = set()  # 已处理文件的集合
    train_files, val_files, test_files = [], [], []

    for label, _ in sorted_labels:
        # 找到包含当前类别的所有图片
        relevant_files = [file for file, labels in image_to_labels.items() if label in labels and file not in mask]
        if not relevant_files:
            continue  # 如果没有未分配的图片，跳过该类别

        # 自定义划分数据集
        temp_train, temp_val, temp_test = custom_split(relevant_files, train_ratio, val_ratio)

        # 更新数据集
        train_files.extend(temp_train)
        val_files.extend(temp_val)
        test_files.extend(temp_test)

        # 将已处理的文件添加到 mask
        mask.update(relevant_files)

    return train_files, val_files, test_files

# 创建目标目录结构
def create_directory_structure(base_dir):
    for sub_dir in ['train', 'val', 'test']:
        os.makedirs(os.path.join(base_dir, sub_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(base_dir, sub_dir, 'labels'), exist_ok=True)

# 复制文件到目标目录
def copy_files(file_list, base_dir, sub_dir, images_dir, labels_dir):
    for file_path in tqdm(file_list, desc=f'Copying {sub_dir} files'):
        image_file = os.path.join(images_dir, file_path)
        label_file = os.path.join(labels_dir, file_path.replace('.jpg', '.txt'))
        
        # 复制图片
        dest_image_dir = os.path.join(base_dir, sub_dir, 'images', os.path.basename(image_file))
        shutil.copy(image_file, dest_image_dir)
        
        # 复制标签
        dest_label_dir = os.path.join(base_dir, sub_dir, 'labels', os.path.basename(label_file))
        shutil.copy(label_file, dest_label_dir)

# 主逻辑
def main():
    images_dir = './images'  # 替换为你的 images 文件夹路径
    labels_dir = './labels'  # 替换为你的 labels 文件夹路径
    base_dir = './stage2_yoloStyle'    # 替换为输出目录路径
    # 统计各类别的数量
    label_counts, image_to_labels = count_labels(labels_dir)
    print("Category counts:", label_counts)

    # 按类别依次划分数据集
    train_files, val_files, test_files = stratified_split_by_category(
        label_counts, image_to_labels, train_ratio=0.8, val_ratio=0.1
    )

    # 创建目录结构
    create_directory_structure(base_dir)

    # 复制文件到目标目录
    copy_files(train_files, base_dir, 'train', images_dir, labels_dir)
    copy_files(val_files, base_dir, 'val', images_dir, labels_dir)
    copy_files(test_files, base_dir, 'test', images_dir, labels_dir)

    print(f'Train set size: {len(train_files)}')
    print(f'Val set size: {len(val_files)}')
    print(f'Test set size: {len(test_files)}')

if __name__ == "__main__":
    main()
