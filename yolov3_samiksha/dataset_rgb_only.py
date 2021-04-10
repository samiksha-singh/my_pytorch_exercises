"""To run inference with darknet weights on yolo sample images, we use this custom dataloader"""
from pathlib import Path

import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class RGBOnly(Dataset):
    def __init__(self, dir_root: Path, transform=None ):

        dir_img = dir_root
        self.list_imgs = self.get_image_list(dir_img)
        num_imgs = len(self.list_imgs)
        if num_imgs < 1:
            raise ValueError(f"No images found in dir: {dir_img}")

        self.transform = transform

    def __len__(self):
        return len(self.list_imgs)

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

        dummy_label = torch.zeros((0, 5))

        if self.transform is not None:
            try:
                image, dummy_bb_target = self.transform((img_numpy, dummy_label))
            except:
                print("Could not apply transform")
                return
        else:
            image = transforms.ToTensor()(img_numpy)

        return image, dummy_bb_target

    @staticmethod
    def get_image_list(dir_img):
        img_path = Path(dir_img)
        list_imgs = sorted(img_path.rglob('*.jpg'))  # this is called a generator
        if len(list_imgs) == 0:
            raise ValueError(f"No images found, {dir_img}")
        return list_imgs

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
    bb_targets = torch.cat(bb_targets, 0)

    return imgs, bb_targets
