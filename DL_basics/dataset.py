import torch
#from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
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
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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

def collate_fn(batch):
    """
    The function creates a list of images and labels into a batch
    The purpose behind using the special function is because the size of our label tensor is not constant because
    we have variable number of bounding boxes per image
    Args:
        batch: List of outputs from each dataset instance in Dataloader

    Returns:
        Tensor: Image
        List[Tensor]: labels

    """
    img_list =[]
    label_list =[]
    for item in batch:
        img_list.append(item[0])
        label_list.append(item[1])
    img_tensor = torch.stack(img_list)

    return img_tensor, label_list


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
    print("size of dataset: ",len(dataset))
    for batch_idx, batch in enumerate(dataset):
        img, label = batch
        img_np = (img.numpy()*255).astype(np.uint8)
        img_np = img_np.transpose((1,2,0))

        label_np = label.numpy()
        img_opencv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        for item in label_np:
            start_point = (item[1], item[2])
            end_point = (item[3], item[4])
            color = (255,0,0)
            thickness = 2
            image = cv2.rectangle(img_opencv, start_point, end_point, color, thickness)
        cv2.imwrite("new_img.png", image)

        if batch_idx > 0:
            break
    #print(label_np)

    training_generator = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True,collate_fn=collate_fn)
    for idx, batch in enumerate(training_generator):
        img , label = batch
        print(img.shape, len(label))
        print(img[0].shape)
        if idx > 0:
            break
if __name__ == "__main__":
    main()




