import wandb
import torch

from utils import utils


def log_bboxes(img_tensor, label_tensor, caption="Train/Prediction", max_images_to_upload=3):
    """Upload images and their bounding boxes to WandB for visualization"""
    # Select first N images only
    img_tensor = img_tensor[:max_images_to_upload]

    img_cat_list = []
    for idx, img in enumerate(img_tensor):
        # select bboxes belonging to this image using image id
        matching_bboxes = []
        for lbl in label_tensor:
            if lbl[0] == idx:
                matching_bboxes.append(lbl)
        label_xywh = torch.stack(matching_bboxes, 0)

        # convert x,y,w,h to xmin,ymin,xmax,ymax
        label_xyxy = utils.xywh_to_xyxy(label_xywh)

        # Convert bbox tensor to wandb Image for logging
        prediction_list = []
        for tensor_ele in label_xyxy:
            bbox_data = {
                "position" : {
                    "minX" : tensor_ele[2].item(),
                    "maxX" : tensor_ele[4].item(),
                    "minY" : tensor_ele[3].item(),
                    "maxY" : tensor_ele[5].item(),
                },
                "class_id": int(tensor_ele[1].item()),
            }
            prediction_list.append(bbox_data)

        image = wandb.Image(
            img,
            boxes={"predictions": {"box_data": prediction_list},
                   "ground_truth": {"box_data": prediction_list}}
        )
        img_cat_list.append(image)

    wandb.log({caption: img_cat_list})
    return
