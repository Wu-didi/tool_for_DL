import os
import json
 
 
class BDD_to_YOLO:
    def __init__(self,writepath = "BDD100K/labels/trains/",select_categorys = [ "car", "bus", "truck"]):
        self.writepath = writepath
        self.bdd100k_width_ratio = 1.0/1280
        self.bdd100k_height_ratio = 1.0/720
        # 可以在select_categorys选择需要的类别，也可以使用全部类别
        self.select_categorys = select_categorys
        # 选择的类别对应的序号
        self.categorys_nums = {
            "car": 0,
            "bus": 0,
            "truck": 0,
        }

        # 可以在select_categorys选择需要的类别，也可以使用全部类别
        # self.select_categorys = ["person", "rider", "car", "bus", "truck", "bike","motor", "traffic light", "traffic sign", "train"]
        # # 选择的类别对应的序号
        # self.categorys_nums = {
        #     "person": 0,
        #     "rider": 1,
        #     "car": 2,
        #     "bus": 3,
        #     "truck": 4,
        #     "bike": 5,
        #     "motor": 6,
        #     "tl_green": 7,
        #     "tl_red": 8,
        #     "tl_yellow": 9,
        #     "tl_none": 10,
        #     "traffic sign": 11,
        #     "train": 12
        # }
 

    def bdd_to_yolo(self, path):
        lines = ""
        with open(path) as fp:
            j = json.load(fp) 
            write = open(self.writepath + "%s.txt" % j["name"], 'w')
 
            for fr in j["frames"]:
 
                for objs in fr["objects"]:
 
                    if objs["category"] in self.select_categorys:
 
                        temp_category=objs["category"]
 
                        if (temp_category == "traffic light"):
 
                            color = objs["attributes"]["trafficLightColor"]
 
                            temp_category="tl_"+color
 
                        idx = self.categorys_nums[temp_category]
 
                        cx = (objs["box2d"]["x1"] + objs["box2d"]["x2"]) / 2.0
 
                        cy = (objs["box2d"]["y1"] + objs["box2d"]["y2"]) / 2.0
 
                        w = objs["box2d"]["x2"] - objs["box2d"]["x1"]
 
                        h = objs["box2d"]["y2"] - objs["box2d"]["y1"]
 
                        if w <= 0 or h <= 0:
 
                            continue
 
                        # 根据图片尺寸进行归一化
 
                        cx, cy, w, h = cx * self.bdd100k_width_ratio, cy * self.bdd100k_height_ratio, w * self.bdd100k_width_ratio, h * self.bdd100k_height_ratio
 
                        line = f"{idx} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n"
 
                        lines += line
 
                if len(lines) != 0:
 
                    write.writelines(lines)
 
                    write.close()
 
                    print("%s has been dealt!" % j["name"])
 
 
if __name__ == "__main__":
 
    bdd_labels_dir = r"E:\python_file\bdd2yolo\oneBDDdataset\label/"
 
    fileList = os.listdir(bdd_labels_dir)
 
    obj = BDD_to_YOLO(writepath='./')
 
    for path in fileList:
 
        filepath = bdd_labels_dir+path
 
        print(path)
 
        obj.bdd_to_yolo(filepath) 