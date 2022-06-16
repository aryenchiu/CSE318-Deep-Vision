import os

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from torchvision import transforms

import _pickle as pickle
import sys
np.set_printoptions(threshold=sys.maxsize)

SEG_LABELS_LIST = [
    {"id": -1, "name": "void",         "rgb_values": [0,0,0], "color": [0, 0, 0]},
    {"id": 0,  "name": "bed",          "rgb_values": [1,0,0], "color": [1, 5, 255]},
    {"id": 1,  "name": "windowpane",   "rgb_values": [2,0,0], "color": [2, 230, 230]},
    {"id": 2,  "name": "cabinet",      "rgb_values": [3,0,0], "color": [3, 5, 255]},
    {"id": 3,  "name": "person",       "rgb_values": [4,0,0], "color": [4, 5, 61]},
    {"id": 4,  "name": "door",         "rgb_values": [5,0,0], "color": [5, 255, 51]},
    {"id": 5,  "name": "table",        "rgb_values": [6,0,0], "color": [6, 6, 82]},
    {"id": 6,  "name": "curtain",      "rgb_values": [7,0,0], "color": [7, 51, 7]},
    {"id": 7,  "name": "chair",        "rgb_values": [8,0,0], "color": [8, 70, 3]},
    {"id": 8,  "name": "car",          "rgb_values": [9,0,0], "color": [9, 102, 200]},
    {"id": 9, "name": "painting",     "rgb_values": [10,0,0], "color": [10, 6, 51]},
    {"id": 10, "name": "sofa",         "rgb_values": [11,0,0], "color": [11, 102, 255]},
    {"id": 11, "name": "shelf",        "rgb_values": [12,0,0], "color": [12, 7, 71]},
    {"id": 12, "name": "mirror",       "rgb_values": [13,0,0], "color": [13, 220, 220]},
    {"id": 13, "name": "armchair",     "rgb_values": [14,0,0], "color": [14, 255, 214]},
    {"id": 14, "name": "seat",         "rgb_values": [15,0,0], "color": [15, 255, 224]},
    {"id": 15, "name": "fence",        "rgb_values": [16,0,0], "color": [16, 184, 6]},
    {"id": 16, "name": "desk",         "rgb_values": [17,0,0], "color": [17, 255, 71]},
    {"id": 17, "name": "wardrobe",     "rgb_values": [18,0,0], "color": [18, 255, 255]},
    {"id": 18, "name": "lamp",         "rgb_values": [19,0,0], "color": [19, 255, 8]},
    {"id": 19, "name": "bathtub",      "rgb_values": [20,0,0], "color": [20, 8, 255]},
    {"id": 20, "name": "railing",      "rgb_values": [21,0,0], "color": [21, 61, 6]},
    {"id": 21, "name": "cushion",      "rgb_values": [22,0,0], "color": [22, 194, 7]},
    {"id": 22, "name": "box",          "rgb_values": [23,0,0], "color": [23, 255, 20]},
    {"id": 23, "name": "column",       "rgb_values": [24,0,0], "color": [24, 8, 41]},
    {"id": 24,  "name": "signboard",          "rgb_values": [25,0,0], "color": [25, 5, 153]},
    {"id": 25,  "name": "chest of drawers",   "rgb_values": [26,0,0], "color": [26, 51, 255]},
    {"id": 26,  "name": "counter",      "rgb_values": [27,0,0], "color": [27, 12, 255]},
    {"id": 27,  "name": "sink",       "rgb_values": [28,0,0], "color": [28, 163, 255]},
    {"id": 28,  "name": "fireplace",         "rgb_values": [29,0,0], "color": [29, 10, 15]},
    {"id": 29,  "name": "refrigerator",        "rgb_values": [30,0,0], "color": [30, 255, 0]},
    {"id": 30,  "name": "stairs",      "rgb_values": [31,0,0], "color": [31, 224, 0]},
    {"id": 31,  "name": "case",        "rgb_values": [32,0,0], "color": [32, 0, 255]},
    {"id": 32,  "name": "pool table",          "rgb_values": [33,0,0], "color": [33, 71, 0]},
    {"id": 33, "name": "pillow",     "rgb_values": [34,0,0], "color": [34, 235, 255]},
    {"id": 34, "name": "screen door",         "rgb_values": [35,0,0], "color": [35, 173, 255]},
    {"id": 35, "name": "bookcase",        "rgb_values": [36,0,0], "color": [36, 255, 245]},
    {"id": 36, "name": "coffee table",       "rgb_values": [37,0,0], "color": [37, 255, 112]},
    {"id": 37, "name": "toilet",     "rgb_values": [38,0,0], "color": [38, 255, 133]},
    {"id": 38, "name": "flower",         "rgb_values": [39,0,0], "color": [39, 0, 0]},
    {"id": 39, "name": "book",        "rgb_values": [40,0,0], "color": [40, 163, 0]},
    {"id": 40, "name": "bench",         "rgb_values": [41,0,0], "color": [41, 255, 0]},
    {"id": 41, "name": "countertop",     "rgb_values": [42,0,0], "color": [42, 143, 255]},
    {"id": 42, "name": "stove",         "rgb_values": [43,0,0], "color": [43, 255, 0]},
    {"id": 43, "name": "palm",      "rgb_values": [44,0,0], "color": [44, 82, 255]},
    {"id": 44, "name": "kitchen island",      "rgb_values": [45,0,0], "color": [45, 255, 41]},
    {"id": 45, "name": "computer",      "rgb_values": [46,0,0], "color": [46, 255, 173]},
    {"id": 46, "name": "swivel chair",          "rgb_values": [47,0,0], "color": [47, 0, 255]},
    {"id": 47, "name": "boat",       "rgb_values": [48,0,0], "color": [48, 255, 0]},
    {"id": 48,  "name": "arcade machine",          "rgb_values": [49,0,0], "color": [49, 92, 0]},
    {"id": 49,  "name": "bus",   "rgb_values": [50,0,0], "color": [50, 0, 245]},
    {"id": 50,  "name": "towel",      "rgb_values": [51,0,0], "color": [51, 0, 102]},
    {"id": 51,  "name": "light",       "rgb_values": [52,0,0], "color": [52, 173, 0]},
    {"id": 52,  "name": "truck",         "rgb_values": [53,0,0], "color": [53, 0, 20]},
    {"id": 53,  "name": "chandelier",        "rgb_values": [54,0,0], "color": [54, 31, 25]},
    {"id": 54,  "name": "awning",      "rgb_values": [55,0,0], "color": [55, 255, 61]},
    {"id": 55,  "name": "streetlight",        "rgb_values": [56,0,0], "color": [56, 71, 255]},
    {"id": 56,  "name": "booth",          "rgb_values": [57,0,0], "color": [57, 57, 204]},
    {"id": 57, "name": "television receiver",     "rgb_values": [58,0,0], "color": [58, 255, 194]},
    {"id": 58, "name": "airplane",         "rgb_values": [59,0,0], "color": [59, 255, 82]},
    {"id": 59, "name": "apparel",        "rgb_values": [60,0,0], "color": [60, 112, 255]},
    {"id": 60, "name": "pole",       "rgb_values": [61,0,0], "color": [61, 0, 255]},
    {"id": 61, "name": "bannister",     "rgb_values": [62,0,0], "color": [62, 122, 255]},
    {"id": 62, "name": "ottoman",         "rgb_values": [63,0,0], "color": [63, 153, 0]},
    {"id": 63, "name": "bottle",        "rgb_values": [64,0,0], "color": [64, 255, 10]},
    {"id": 64, "name": "van",         "rgb_values": [65,0,0], "color": [65, 255, 0]},
    {"id": 65, "name": "ship",     "rgb_values": [66,0,0], "color": [66, 235, 0]},
    {"id": 66, "name": "fountain",         "rgb_values": [67,0,0], "color": [67, 184, 170]},
    {"id": 67, "name": "washer",      "rgb_values": [68,0,0], "color": [68, 0, 255]},
    {"id": 68, "name": "plaything",      "rgb_values": [69,0,0], "color": [69, 0, 31]},
    {"id": 69, "name": "stool",      "rgb_values": [70,0,0], "color": [70, 214, 255]},
    {"id": 70, "name": "barrel",          "rgb_values": [71,0,0], "color": [71, 0, 112]},
    {"id": 71, "name": "basket",       "rgb_values": [72,0,0], "color": [72, 255, 0]},
    {"id": 72,  "name": "bag",          "rgb_values": [73,0,0], "color": [73, 184, 160]},
    {"id": 73,  "name": "minibike",   "rgb_values": [74,0,0], "color": [74, 0, 255]},
    {"id": 74,  "name": "oven",      "rgb_values": [75,0,0], "color": [75, 255, 0]},
    {"id": 75,  "name": "ball",       "rgb_values": [76,0,0], "color": [76, 0, 163]},
    {"id": 76,  "name": "food",         "rgb_values": [77,0,0], "color": [77, 204, 0]},
    {"id": 77,  "name": "step",        "rgb_values": [78,0,0], "color": [78, 0, 143]},
    {"id": 78,  "name": "trade name",      "rgb_values": [79,0,0], "color": [79, 255, 0]},
    {"id": 79,  "name": "microwave",        "rgb_values": [80,0,0], "color": [80, 0, 235]},
    {"id": 80,  "name": "pot",          "rgb_values": [81,0,0], "color": [81, 0, 255]},
    {"id": 81, "name": "animal",     "rgb_values": [82,0,0], "color": [82, 0, 122]},
    {"id": 82, "name": "bicycle",         "rgb_values": [83,0,0], "color": [83, 245, 0]},
    {"id": 83, "name": "dishwasher",        "rgb_values": [84,0,0], "color": [84, 255, 0]},
    {"id": 84, "name": "screen",       "rgb_values": [85,0,0], "color": [85, 204, 255]},
    {"id": 85, "name": "sculpture",     "rgb_values": [86,0,0], "color": [86, 255, 0]},
    {"id": 86, "name": "hood",         "rgb_values": [87,0,0], "color": [87, 153, 255]},
    {"id": 87, "name": "sconce",        "rgb_values": [88,0,0], "color": [88, 41, 255]},
    {"id": 88, "name": "vase",         "rgb_values": [89,0,0], "color": [89, 255, 204]},
    {"id": 89, "name": "traffic light",     "rgb_values": [90,0,0], "color": [90, 0, 255]},
    {"id": 90, "name": "tray",         "rgb_values": [91,0,0], "color": [91, 255, 0]},
    {"id": 91, "name": "ashcan",      "rgb_values": [92,0,0], "color": [92, 0, 255]},
    {"id": 92, "name": "fan",      "rgb_values": [93,0,0], "color": [93, 245, 255]},
    {"id": 93, "name": "plate",      "rgb_values": [94,0,0], "color": [94, 255, 184]},
    {"id": 94, "name": "monitor",          "rgb_values": [95,0,0], "color": [95, 92, 255]},
    {"id": 95, "name": "bullentin board",       "rgb_values": [96,0,0], "color": [96, 255, 0]},
    {"id": 96,  "name": "radiator",          "rgb_values": [97,0,0], "color": [97, 214, 0]},
    {"id": 97,  "name": "glass",   "rgb_values": [98,0,0], "color": [98, 194, 194]},
    {"id": 98,  "name": "clock",      "rgb_values": [99,0,0], "color": [99, 255, 0]},
    {"id": 99,  "name": "flag",       "rgb_values": [100,0,0], "color": [100, 0, 255]}
]

def label_img_to_rgb(label_img):
    label_img = np.squeeze(label_img)
    labels = np.unique(label_img)
    label_infos = [l for l in SEG_LABELS_LIST if l['id'] in labels]

    label_img_rgb = np.array([label_img,
                              label_img,
                              label_img]).transpose(1,2,0)
    for l in label_infos:
        mask = label_img == l['id']
        label_img_rgb[mask] = l['color']

    return label_img_rgb.astype(np.uint8)

class SegmentationData(data.Dataset):

    def __init__(self, image_paths_file):
        self.root_dir_name = os.path.dirname(image_paths_file)

        with open(image_paths_file) as f:
            self.image_names = f.read().splitlines()

    def __getitem__(self, key):
        if isinstance(key, slice):
            # get the start, stop, and step from the slice
            return [self[ii] for ii in range(*key.indices(len(self)))]
        elif isinstance(key, int):
            # handle negative indices
            if key < 0:
                key += len(self)
            if key < 0 or key >= len(self):
                raise IndexError("The index (%d) is out of range." % key)
            # get the data from direct index
            return self.get_item_from_index(key)
        else:
            raise TypeError("Invalid argument type.")

    def __len__(self):
        return len(self.image_names)

    def get_item_from_index(self, index):
        # print(" index: ", index)
        to_tensor = transforms.ToTensor()
        img_id = self.image_names[index].replace('.jpg', '')

        # Load image 
        img = Image.open(os.path.join(self.root_dir_name,
                                      'images',
                                      img_id + '.jpg')).convert('RGB')

        # center_crop = transforms.CenterCrop(240)
        resize = transforms.Resize((256, 256))
        img = resize(img)
        # img = center_crop(img)
        img = to_tensor(img)

        # Load target
        target = Image.open(os.path.join(self.root_dir_name,
                                         'annotations_instance',
                                         img_id + '.png')).convert('RGB')
        r, g, b = target.split()
        empty = Image.new("L", target.size, "black")
        target = Image.merge("RGB", (r, empty, empty))
        resize = transforms.Resize((256, 256), transforms.InterpolationMode.NEAREST)
        target = resize(target)
        # target = center_crop(target)
        target = np.array(target, dtype=np.int64)
        target_labels = target[..., 0]
        
        for label in SEG_LABELS_LIST:
            mask = np.all(target == label['rgb_values'], axis=2)
            target_labels[mask] = label['id']

        target_labels = torch.from_numpy(target_labels.copy())
        return img, target_labels
