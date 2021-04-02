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
from transforms import DEFAULT_TRANSFORMS


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
        img_numpy = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        f_label = self.list_labels[index]
        label = self.read_label(f_label)
        label_numpy = np.array(label, dtype=np.float32)

        if self.transform is not None:
            try:
                image, bb_target = self.transform((img_numpy, label_numpy))
            except:
                print("Could not apply transform")
                return
        else:
            image = transforms.ToTensor()(img_numpy)
            label = torch.tensor(label_numpy).unsqueeze(0)  # Add channels dimension
            _, h, w = image.shape

            # Convert xyxy (min/max) to xywh
            x_min = label[:,:,1]
            y_min = label[:,:,2]
            x_max = label[:,:,3]
            y_max = label[:,:,4]
            bb_target = torch.zeros_like(label)
            bb_target[:,:,1] = (x_min + x_max) / 2
            bb_target[:,:,2] = (y_min + y_max) / 2
            bb_target[:,:,3] = x_max - x_min
            bb_target[:,:,4] = y_max - y_min

            # to convert absolute coordinated into relative coordinates
            bb_target[:, :, [1, 3]] /= w
            bb_target[:, :, [2, 4]] /= h

        return image, bb_target

    def read_label(self, label):
        ele_obj_list = []
        tree = ET.parse(label)
        root = tree.getroot()
        for _object in root.findall('object'):
            list_obj = []
            ele_obj = _object.find('name').text
            list_obj.append(ele_obj)

            bbox = _object.find('bndbox')
            for child in bbox:
                # Iterate over xmin/xmax/ymin/ymax within the xml file
                list_obj.append(int(child.text))
            ele_obj_list.append(list_obj)

        # change the sting class value to integer value for it to convert into a tensor
        label_list = []
        for item in ele_obj_list:
            class_str = item[0]
            class_int = self.class_dict[class_str]
            item[0] = class_int
            label_list.append(item)
        return label_list

    @staticmethod
    def get_image_list(dir_img):
        img_path = Path(dir_img)
        list_imgs = sorted(img_path.rglob('*.jpg'))  # this is called a generator
        if len(list_imgs) == 0:
            raise ValueError(f"No images found, {dir_img}")
        return list_imgs

    @staticmethod
    def get_label_list(dir_label):
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


def draw_bbox(img, label):
    img_np = (img.numpy() * 255).astype(np.uint8)
    img_np = img_np.transpose((1, 2, 0)) #to change the order of channel
    height, width, _ = img_np.shape

    # Remove channels dim from label
    label = label.squeeze(0)

    # Convert to absolute coords
    label[:, [1, 3]] *= width
    label[:, [2, 4]] *= height

    # Convert to int
    label = label.round().int()

    # Convert xywh to xyxy (min/max)
    x = label[:, 1]
    y = label[:, 2]
    w = label[:, 3]
    h = label[:, 4]
    label_xyxy = torch.zeros_like(label)
    label_xyxy[:, 1] = (x - (w/2))
    label_xyxy[:, 2] = (y - (h/2))
    label_xyxy[:, 3] = (x + (w/2))
    label_xyxy[:, 4] = (y + (h/2))
    label_xyxy[:, 0] = label[:, 0]

    label_np = label_xyxy.numpy()
    img_opencv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    for box in label_np:
        start_point = (box[1], box[2])
        end_point = (box[3], box[4])
        color = (255, 0, 0)
        thickness = 2
        image = cv2.rectangle(img_opencv, start_point, end_point, color, thickness)
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_img", type=Path, required=True)
    parser.add_argument("--dir_label", type=Path,  required=True)
    args = parser.parse_args()

    dir_img = args.dir_img
    dir_label = args.dir_label

    if not dir_img.is_dir():
        raise ValueError(f"Not a directory {dir_img}")

    dataset = PascalVOC(dir_img,dir_label, transform=DEFAULT_TRANSFORMS)
    print("size of dataset: ", len(dataset))

    training_generator = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

    for idx, batch in enumerate(training_generator):
        images, labels = batch

        img_bbox = []
        for img, label in zip(images, labels):
            # single image and its label
            img = draw_bbox(img, label)
            img_bbox.append(img)

        concat_img = np.concatenate(img_bbox, axis=1)
        cv2.imwrite(f"dataset_sample_batch_{idx}.jpg", concat_img)

        if idx >= 1:
            break


if __name__ == "__main__":
    main()




