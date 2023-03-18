import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

label_dir = './mask2/labels/train'
img_dir = './mask2/images/train'
aspect_ratio_bins = np.arange(0, 5, 0.05)

aspect_ratios = []
for label_file in os.listdir(label_dir):
    with open(os.path.join(label_dir, label_file), 'r') as f:
        for line in f.readlines():
            label = line.strip().split()
            class_id, x_center, y_center, width, height = map(float, label[0:5])
            img_file = os.path.join(img_dir, os.path.splitext(label_file)[0] + '.jpg')
            print(img_file)
            img = cv2.imread(img_file)
            img_height, img_width, _ = img.shape
            bbox_height, bbox_width = height * img_height, width * img_width
            if bbox_width == 0:
                continue
            aspect_ratio = bbox_height / bbox_width
            aspect_ratios.append(aspect_ratio)

aspect_ratios = np.array(aspect_ratios)
hist, bin_edges = np.histogram(aspect_ratios, bins=aspect_ratio_bins)
print(hist)
print(bin_edges)
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(bin_edges[:-1], hist/sum(hist), width=0.1)
ax.set_xlabel('Aspect Ratio')
ax.set_ylabel('Probability')
ax.set_title('Aspect Ratio Distribution')
plt.show()
