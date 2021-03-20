from __future__ import print_function, division
import torch
#from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import parseXml
import savepath
import cv2
from pathlib import Path
import argparse

class PascalVOC(Dataset):
    def __init__(self, dir_img, dir_label, transform=None ):

        self.list_imgs = self.get_image_list(dir_img)
        self.list_labels = self.get_label_list(dir_label)
        self.transform = transform

        self.class_dict = {
            "background": 0,
            "aeroplane": 1,
            "bicycle": 2,
            "bird": 3,
            "boat": 4,
            "bottle": 5,
            "bus": 6,
            "car": 7,
            "cat": 8,
            "chair": 9,
            "cow": 10,
            "diningtable": 11,
            "dog": 12,
            "horse": 13,
            "motorbike": 14,
            "person": 15,
            "pottedplant": 16,
            "sheep": 17,
            "sofa": 18,
            "train": 19,
            "tvmonitor": 20
        }

    def __len__(self):
        return len(self.list_labels)

    def __getitem__(self, index):
        f_img = self.list_imgs[index]
        img = cv2.imread(str(f_img))
        img_tensor = transforms.ToTensor()(img)


        f_label = self.list_labels[index]
        label = self.read_label(f_label)
        label_tensor = torch.tensor(label)

        return img_tensor, label_tensor

    def read_label(self,label):
        ele_obj_list = []
        tree = ET.parse(label)
        root = tree.getroot()
        for _object in root.findall('object'):
            # print(_object)
            list_obj = []
            ele_obj = _object.find('name').text
            list_obj.append(ele_obj)
            # print(list_obj)

            bbox = _object.find('bndbox')
            for child in bbox:
                list_obj.append(int(child.text))
                # print(list_obj)
            ele_obj_list.append(list_obj)

        # change the sting class value to integer value for it to convert into a tensor
        label_list = []
        for item in ele_obj_list:
            class_str = item[0]
            class_int = self.class_dict[class_str]
            item[0] = class_int
            label_list.append(item)
        #print(ele_obj_list)
        return label_list

    def get_image_list(self, dir_img):
        img_path = Path(dir_img)
        list_imgs = sorted(img_path.rglob('*.jpg'))  # this is called a generator
        if len(list_imgs) == 0:
            raise ValueError(f"No images found, {dir_img}")
        return list_imgs

    def get_label_list(self,dir_label):
        label_path = Path(dir_label)
        list_labels = sorted(label_path.rglob('*.xml'))
        if len(list_labels) == 0:
            raise ValueError(f"No images found, {dir_label}")
        return list_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_img", type=Path, required=True)
    parser.add_argument("--dir_label", type=Path,  required=True)
    args = parser.parse_args()

    dir_img = args.dir_img
    dir_label = args.dir_label

    if not dir_img.is_dir():
        raise ValueError(f"Not a directory {dir_img}")

    dataset = PascalVOC(dir_img,dir_label)
    print(len(dataset))
    a, b = dataset[0]
    print("hello")
    print(type(a), type(b))

if __name__ == "__main__":
    main()




