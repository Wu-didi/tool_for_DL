# tool_for_DL  深度学习相关工具
主要深度学习数据处理等内容

## Analysis_dataset文件夹
目标检测数据集分析
统计目标检测数据集中目标框的为位置，用热力图展示
统计目标框的长宽比和高度用直方图展示

## frame2video.py
将连续帧转化为视频（mp4格式）\n

 python frames2video.py --image_folder ./data/source/train/ --video_name origin_result.mp4 --resize --resize-shape 728 256

## copy_images_from_folders.py
从众多文件夹中提取出想要的图片，采用递归的思想去遍历所有的文件。**该脚本是提取到color文件夹下，以.png结尾的图片。后期可以根据需要进行修改。**
