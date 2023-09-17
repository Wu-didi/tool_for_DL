import cv2
import shutil
import os
from tqdm import tqdm
sets = ["train","valid"]
# data_path  = ''
img_path = './training/image_2'
label_path = 'Annotations/'
for image_set in sets:
    image_ids = open(r'imageset/%s.txt'%(image_set)).read().strip().split()
    for image_path in tqdm(image_ids):
#         print(image_path)
        #
        one_img_path = os.path.join(img_path,image_path+'.png')
        one_label_path = os.path.join(label_path,'trian'+image_path+'.txt')
        shutil.copyfile(one_img_path,image_set+"//"+'images'+'/'+'%s.png'%(image_path))
        shutil.copy(one_label_path,image_set+"//"+'labels'+'/'+'%s.txt'%(image_path))