import glob

txt_list = glob.glob(r'D:\BaiduNetdiskWorkspace\python_file2\frustum_pointnets_pytorch-master\dataset\KITTI\object\training\label_2/*.txt') # 存储Labels文件夹所有txt文件路径
txt_list = txt_list[:10]
print(txt_list)
print(len(txt_list))