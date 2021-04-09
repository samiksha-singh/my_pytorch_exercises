import wandb
import torch

from utils import utils


def log_bboxes(img_tensor, label_tensor, outputs_list, caption="Train/Prediction", max_images_to_upload=3):
    """Upload images and their bounding boxes to WandB for visualization

    Args:
        img_tensor (torch.Tensor): Shape: [b, c, h, w]. Batch of images.

        label_tensor (torch.Tensor): Shape: [N, 6] (img_id, cls_id, x1, y1, x2, y2). Bboxes for all the images.
            Label for each image can be identified using the image id. Absolute coords for bbox.

        outputs_list (List[Tensor]): Output of the model (in eval mode, after non-max suppression). List of
            bounding boxes in shape [N, 6] (x1, y1, x2, y2, conf, cls). Absolute coords for bbox.
    """
    # Select first N images only
    img_tensor = img_tensor[:max_images_to_upload]

    img_cat_list = []
    for idx, img in enumerate(img_tensor):
        # select label bboxes belonging to this image using image id
        matching_bboxes = []
        for lbl in label_tensor:
            if lbl[0] == idx:
                matching_bboxes.append(lbl)
        label_xyxy = torch.stack(matching_bboxes, 0)

        # convert x,y,w,h to xmin,ymin,xmax,ymax
        # label_xyxy = utils.xywh_to_xyxy(label_xywh)

        # Create dict for logging preds to wandb.
        # Convert outputs to relative coords
        img_size = img.shape[0]
        label_xyxy[-4:] /= float(img_size)
        label_list = []
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
            label_list.append(bbox_data)

        # Create dict for logging labels to wandb.
        output = outputs_list[idx]
        # Convert outputs to relative coords
        img_size = img.shape[0]
        output[:4] /= float(img_size)

        prediction_list = []
        for tensor_ele in output:
            bbox_data = {
                "position": {
                    "minX": tensor_ele[0].item(),
                    "maxX": tensor_ele[1].item(),
                    "minY": tensor_ele[2].item(),
                    "maxY": tensor_ele[3].item(),
                },
                "class_id": int(tensor_ele[1].item()),
            }
            prediction_list.append(bbox_data)

        # Construct WandB image obj for logging.
        image = wandb.Image(
            img,
            boxes={"predictions": {"box_data": prediction_list},
                   "ground_truth": {"box_data": label_list}}
        )
        img_cat_list.append(image)

    wandb.log({caption: img_cat_list}, commit=False)
    return
