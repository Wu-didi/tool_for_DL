import os

# 标签映射字典
label_map = {
    "veh_normal_signal": 0,
    "per_normal_signal": 1,
    "veh_multi_signal": 2,
    "per_multi_signal": 3,
    "veh_abnormal_signal": 4,
    "per_abnormal_signal": 5,
    "red": 6,
    "yellow": 7,
    "green": 8,
}

# 统计类别数量
def count_labels(labels_dir):
    label_count = {label: 0 for label in label_map.values()}
    
    for filename in os.listdir(labels_dir):
        if filename.endswith('.txt'):
            with open(os.path.join(labels_dir, filename), 'r') as f:
                for line in f:
                    label = int(line.split()[0])  # 获取类别编号
                    if label in label_count:
                        label_count[label] += 1
    
    return label_count

labels_dir = './stage2_yoloStyle/val/labels'  # 替换为实际路径
label_count = count_labels(labels_dir)
print(label_count)
