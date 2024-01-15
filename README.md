# tool_for_DL  深度学习相关工具
主要深度学习数据处理等内容

## 目标检测数据集分析
Analysis_dataset文件夹
统计目标检测数据集中目标框的为位置，用热力图展示
统计目标框的长宽比和高度用直方图展示

## frame2video.py
将连续帧转化为视频（mp4格式）
'''python
 python frames2video.py --image_folder ./data/source/train/ --video_name origin_result.mp4 --resize --resize-shape 728 256
'''

