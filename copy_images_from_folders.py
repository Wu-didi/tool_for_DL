'''
查找文件夹内，相应的图片，以.png结尾的图片，并将其重命名保存。
'''
import os
import shutil
import argparse
import tqdm

def find_and_rename_pngs(source_folder, destination_folder,target_folder):
    idx = 0
    for root, dirs, files in tqdm.tqdm(os.walk(source_folder)):
        if 'Color' in dirs:
            out_folder = os.path.join(root, 'Color')
            for file in os.listdir(out_folder):
                if file.endswith('.png'):
                    source_path = os.path.join(out_folder, file)
                    destination_path = os.path.join(destination_folder, str(idx)+".png")
                    idx += 1
                    shutil.copy2(source_path, destination_path)
                    print(f'Renamed and copied: {file}')

def main():
    parser = argparse.ArgumentParser(description='Find and rename PNG files from source folder to destination folder.')
    parser.add_argument('--source', type=str, help='Source folder path', default='./output')
    parser.add_argument('--destination', type=str, help='Destination folder path', default='./trainA')

    args = parser.parse_args()

    source_folder = args.source
    destination_folder = args.destination

    # Create destination folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Execute operation
    find_and_rename_pngs(source_folder, destination_folder)

if __name__ == "__main__":
    main()
