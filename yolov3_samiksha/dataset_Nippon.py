import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from transforms import get_transforms
from utils import utils


class NipponDataset(Dataset):
    def __init__(self, dir_root: Path, transform=None ):

        dir_img = dir_root / Path("images")
        dir_label = dir_root / Path("annotations")
        self.list_imgs = self.get_image_list(dir_img)
        self.list_labels = self.get_label_list(dir_label)
        num_imgs = len(self.list_imgs)
        num_labels = len(self.list_labels)
        if num_imgs != num_labels:
            raise ValueError(f"Num of images ({num_imgs}) is not equal to num of labels ({num_labels}) "
                f"in dataset.\n  dir_img: {dir_img}\ndir_label: {dir_label}")

        self.transform = transform
        self.class_dict_reverse = {
            0: "car",
            1: "5_dashes",
            2: "dotted_lanes",
        }

        self.class_dict = {
            "car":0,
            "5_dashes":1,
            "dotted_lanes":2,
        }

    def __len__(self):
        return len(self.list_labels)

    def __getitem__(self, index):
        """Return image, label
        Args:
            index: index of img/label to extract

        Returns:
            Tensor: image. Shape=[H, W, 3]
            Tensor: label. Shape=[N, 5] (cls_id, x, y, w, h). Bounding boxes in format x,y,w,h in relative coords.

        Notes:
            The collate fn converts label to [N, 6] shape, by concatenating all the labels along the 0th axis. It also
            adds a new element to 1st axis, for image id. The image id is used to select the labels that belong to a
            particular image.
        """
        f_img = self.list_imgs[index]
        img = cv2.imread(str(f_img))
        img_numpy = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        f_label = self.list_labels[index]
        label_numpy = self.read_label(f_label)

        if self.transform is not None:
            try:
                image, bb_target = self.transform((img_numpy, label_numpy))
            except:
                print("Could not apply transform")
                return
        else:
            image = transforms.ToTensor()(img_numpy)
            label = torch.tensor(label_numpy)
            _, h, w = image.shape

            # Convert xyxy (min/max) to xywh
            y_min = label[:, 2]
            x_min = label[:, 1]
            x_max = label[:, 3]
            y_max = label[:, 4]
            bb_target = torch.zeros_like(label)
            bb_target[:, 1] = (x_min + x_max) / 2
            bb_target[:, 2] = (y_min + y_max) / 2
            bb_target[:, 3] = x_max - x_min
            bb_target[:, 4] = y_max - y_min

            # to convert absolute coordinated into relative coordinates
            bb_target[:, [1, 3]] /= w
            bb_target[:, [2, 4]] /= h

        return image, bb_target

    def read_label(self, f_label):
        label_np = np.loadtxt(f_label, delimiter=",")

        # If single bbox is present, shape will be (5,) instead (1, 5)
        if len(label_np.shape) == 1:
            label_np = np.expand_dims(label_np, axis=0)

        return label_np

    @staticmethod
    def get_image_list(dir_img):
        img_path = Path(dir_img)
        list_imgs = sorted(img_path.rglob('*.png'))  # this is called a generator
        if len(list_imgs) == 0:
            raise ValueError(f"No images found, {dir_img}")
        return list_imgs

    @staticmethod
    def get_label_list(dir_label):
        label_path = Path(dir_label)
        list_labels = sorted(label_path.rglob('*.txt'))
        if len(list_labels) == 0:
            raise ValueError(f"No labels found, {dir_label}")
        return list_labels


def collate_fn(batch):
    """
    The function creates a batch of images and labels.
    We need to use a custom collate func because we cannot directly create a batch of label tensors, since the number
    of bounding boxes per image is different (cannot have tensor of shape BxNx5, because N is variable per image).

    Args:
        batch: List of outputs from each dataset instance in Dataloader

    Returns:
        Tensor: Image. Shape=[b, c, h, w]
        Tensor: labels. Shape=[N, 6] (img_id, cls_id, x, y, w, h). img_id is in range [0, batch_size].
    """
    batch = [data for data in batch if data is not None]
    imgs, bb_targets = list(zip(*batch))

    # Resize images to input shape
    imgs = torch.stack([img for img in imgs])

    # Add sample index to targets.
    # If each label is shape (N, 5), we concatenate along the 0th axis. To distinguish bboxes of different images,
    # we add an image index to the 1st axis.
    for i, boxes in enumerate(bb_targets):
        boxes[:, 0] = i

    bb_targets_cat = torch.cat(bb_targets, 0)

    return imgs, bb_targets_cat


def draw_bbox(img, label):
    """
    Draw bbox on the img for visualization to check correctness of data

    Args:
        img (torch.Tensor): RGB image. Shape: (3, H, W)
        label (torch.Tensor): Bounding boxes corresponding to this rgb image.
            Shape: [N, 6] (img_id, cls_id, xmin, ymin, w, h) -> relative coords.

    Returns:
        numpy.ndarray: RGB image with rectangles drawn around the objects
    """
    img_np = (img.numpy() * 255).astype(np.uint8)
    img_np = img_np.transpose((1, 2, 0)) #to change the order of channel
    height, width, _ = img_np.shape

    # Convert to absolute coords
    label[:, [2, 4]] *= width
    label[:, [3, 5]] *= height

    # Convert to int
    label = label.round().int()

    # Convert xywh to xyxy (min/max)
    label_xyxy = utils.xywh_to_xyxy(label)

    label_np = label_xyxy.numpy()
    img_opencv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    for box in label_np:
        start_point = (box[2], box[3])
        end_point = (box[4], box[5])
        color = (255, 0, 0)
        thickness = 2
        image = cv2.rectangle(img_opencv, start_point, end_point, color, thickness)
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_root", type=Path, required=True,
                        help="Root directory for train. Contains the VOCdevkit dir")
    args = parser.parse_args()

    dir_root = args.dir_root
    if not dir_root.is_dir():
        raise ValueError(f"Not a directory: {dir_root}")

    dataset = NipponDataset(dir_root, transform=get_transforms(img_size=416))
    print("Size of dataset: ", len(dataset))

    training_generator = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

    for idx, batch in enumerate(training_generator):
        images, labels = batch

        img_bbox = []
        for img_idx, img in enumerate(images):
            # single image and its label
            label_xyxy = utils.select_bbox_from_img_id(labels, img_idx)

            img = draw_bbox(img, label_xyxy)
            img_bbox.append(img)

        concat_img = np.concatenate(img_bbox, axis=1)
        cv2.imwrite(f"output/dataset_sample_batch_{idx}.jpg", concat_img)

        if idx >= 1:
            break


if __name__ == "__main__":
    main()
