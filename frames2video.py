import cv2  
import os  
import argparse  
from tqdm import tqdm

def convert_images_to_video(image_folder, video_name, fps=5, resize_shape=None):  
    """Convert a sequence of images to a video."""  
    # 获取文件夹中的所有.png图片  
    images = [img for img in os.listdir(image_folder) if img.lower().endswith(".png")]  
  
    if not images:  
        print(f"No .png images found in the folder: {image_folder}")  
        return  
  
    # 检查是否需要重新调整图像大小  
    if resize_shape:  
        resize_width, resize_height = resize_shape  
        print(f"Resizing images to {resize_width}x{resize_height}")  
  
    # 读取第一张图片来确定视频的尺寸  
    frame = cv2.imread(os.path.join(image_folder, images[0]))  
    height, width, layers = frame.shape  
  
    # 定义视频编码器和输出视频对象  
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用MP4V编码器  
    video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))  
  
    # 重新调整图像大小（如果指定了形状）  
    if resize_shape:  
        for image in tqdm(images):  
            resized_image = cv2.resize(cv2.imread(os.path.join(image_folder, image)), (resize_width, resize_height))  
            video.write(resized_image)  
    else:  
        # 将所有图片写入视频  
        for image in tqdm(images):  
            video.write(cv2.imread(os.path.join(image_folder, image)))  
  
    # 释放视频对象  
    video.release()  
    print(f"Video created: {video_name}")  
  
if __name__ == "__main__":  
    parser = argparse.ArgumentParser(description='Convert a sequence of images to a video.')  
    parser.add_argument('--image_folder', type=str, help='Path to the folder containing the images.')  
    parser.add_argument('--video_name', type=str, help='Name of the output video file.')  
    parser.add_argument('--fps', type=int, default=5, help='Frames per second for the video (default is 5).')  
    parser.add_argument('--resize', action='store_true', help='Resizes the images to the specified shape.')  
    parser.add_argument('--resize-shape', type=int, nargs=2, metavar=('WIDTH', 'HEIGHT'), help='Width and height of the resized image.')  
    args = parser.parse_args()  
  
    if not args.image_folder or not args.video_name:  
        print("Please provide both the image folder and video name.")  
        parser.print_help()  
        exit()  
    elif args.resize and not args.resize_shape:  
        print("When using --resize, please specify the width and height with --resize-shape.")  
        parser.print_help()  
        exit()  
    else:  
        convert_images_to_video(args.image_folder, args.video_name, args.fps, args.resize_shape)