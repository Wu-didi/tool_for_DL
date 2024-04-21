# import json
# import cv2
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches

# def visualize_single_image(image_path, json_path):
#     # Load image
#     image = cv2.imread(image_path)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#     # Load JSON
#     with open(json_path, 'r') as f:
#         data = json.load(f)

#     # Visualize image
#     plt.figure(figsize=(10, 10))
#     plt.imshow(image)

#     # Visualize annotations
#     for frame in data["frames"]:
#         for obj in frame["objects"]:
#             x1 = int(obj["box2d"]["x1"])
#             y1 = int(obj["box2d"]["y1"])
#             x2 = int(obj["box2d"]["x2"])
#             y2 = int(obj["box2d"]["y2"])
#             category = obj["category"]

#             rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none')
#             plt.gca().add_patch(rect)
#             plt.text(x1, y1 - 5, category, color='r')

#     plt.axis('off')
#     plt.show()

# # Example usage
# json_path =  r'E:\python_filev2\yolov5-6.0\datasets\train\BDD_style_labels\000000.json'
# image_path = r'E:\python_filev2\yolov5-6.0\datasets\train\images\000000.png'
# visualize_single_image(image_path, json_path)


import json
import cv2

def visualize_single_image(image_path, json_path):
    # Load image
    image = cv2.imread(image_path)

    # Load JSON
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Draw annotations on image
    for frame in data["frames"]:
        for obj in frame["objects"]:
            x1 = int(obj["box2d"]["x1"])
            y1 = int(obj["box2d"]["y1"])
            x2 = int(obj["box2d"]["x2"])
            y2 = int(obj["box2d"]["y2"])
            category = obj["category"]

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, category, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display image
    cv2.imshow("Image with BDD Annotations", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
json_path =  r'E:\python_filev2\yolov5-6.0\datasets\train\BDD_style_labels\000000.json'
image_path = r'E:\python_filev2\yolov5-6.0\datasets\train\images\000000.png'

visualize_single_image(image_path, json_path)
