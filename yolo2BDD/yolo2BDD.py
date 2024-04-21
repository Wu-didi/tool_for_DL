import os
import json
import tqdm

def convert_yolo_to_bdd(yolo_labels_path, image_width, image_height):
    bdd_annotations = []

    with open(yolo_labels_path, 'r') as f:
        for line in f:
            line = line.strip().split()
            if len(line) < 5:
                continue
            label_id = int(line[0])
            x_center = float(line[1])
            y_center = float(line[2])
            width = float(line[3])
            height = float(line[4])

            x_min = max(0, (x_center - width / 2) * image_width)
            y_min = max(0, (y_center - height / 2) * image_height)
            x_max = min(image_width, (x_center + width / 2) * image_width)
            y_max = min(image_height, (y_center + height / 2) * image_height)

            bdd_annotation = {
                "category": f"name{label_id}",
                "id": 0,
                "attributes": {
                    "occluded": False,
                    "truncated": False
                },
                "box2d": {
                    "x1": x_min,
                    "y1": y_min,
                    "x2": x_max,
                    "y2": y_max
                }
            }

            bdd_annotations.append(bdd_annotation)

    return bdd_annotations

def convert_folder_to_bdd(yolo_folder_path, output_folder_path, image_width, image_height):
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    for file_name in tqdm.tqdm(os.listdir(yolo_folder_path)):
        if file_name.endswith(".txt"):
            yolo_labels_path = os.path.join(yolo_folder_path, file_name)
            output_json_path = os.path.join(output_folder_path, file_name.replace(".txt", ".json"))

            bdd_annotations = convert_yolo_to_bdd(yolo_labels_path, image_width, image_height)
            save_bdd_json(bdd_annotations, output_json_path,file_name)

def save_bdd_json(bdd_annotations, output_json_path, file_name):
    file_name_noend = file_name.split('.')[0]
    data = {
        "name": str(file_name_noend),
        "frames": [
            {
                "timestamp": 10000,
                "objects": bdd_annotations
            }
        ],
        "attributes": {
            "weather": "clear",
            "scene": "city street",
            "timeofday": "daytime"
        }
    }

    with open(output_json_path, 'w') as f:
        json.dump(data, f, indent=4)

# Example usage
yolo_folder_path = r"E:\python_filev2\yolov5-6.0\datasets\val\labels"
output_folder_path = r"E:\python_filev2\yolov5-6.0\datasets\val\BDD_style_labels"

image_width = 1920  # 替换为你的图像宽度
image_height = 1080  # 替换为你的图像高度
convert_folder_to_bdd(yolo_folder_path, output_folder_path, image_width, image_height)
