import wandb
import torch

from utils import utils


def log_bboxes(img_tensor,
               label_tensor,
               outputs_list,
               class_dict,
               wandb_heading="Test/Predictions",
               max_images_to_upload=4):
    """Upload images and their bounding boxes to WandB for visualization

    Args:
        img_tensor (torch.Tensor): Shape: [b, c, h, w]. Batch of images.

        label_tensor (torch.Tensor): Shape: [N, 6] (img_id, cls_id, x1, y1, x2, y2). Bboxes for all the images.
            Label for each image can be identified using the image id. Absolute coords for bbox.

        outputs_list (List[Tensor]): Output of the model (in eval mode, after non-max suppression). List of
            bounding boxes in shape [N, 6] (x1, y1, x2, y2, conf, cls). Absolute coords for bbox.

        class_dict (Dict): Dict mapping ints to class names (strings)

        wandb_heading (str): The heading under which the images will be uploaded to wandb

        max_images_to_upload (int): The max num of images to upload at a time to wandb
    """
    # Select first N images only
    img_tensor = img_tensor[:max_images_to_upload]

    img_cat_list = []
    for idx, img in enumerate(img_tensor):
        # select label bboxes belonging to this image using image id
        label_xyxy = utils.select_bbox_from_img_id(label_tensor, idx)

        _, height, width = img.shape
        # Create dict for logging labels to wandb.
        label_xyxy[:, -4:] /= float(height)  # Convert to relative coords
        label_list = []
        for tensor_ele in label_xyxy:
            cls_id = int(tensor_ele[1].item())
            bbox_data = {
                "position" : {
                    "minX" : tensor_ele[2].item(),
                    "maxX" : tensor_ele[4].item(),
                    "minY" : tensor_ele[3].item(),
                    "maxY" : tensor_ele[5].item(),
                },
                "class_id": cls_id,
                # "box_caption": f"{class_dict[cls_id]}",
            }
            label_list.append(bbox_data)

        # Create dict for logging preds to wandb.
        output = outputs_list[idx]
        output[:, :4] /= float(height)  # Convert outputs to relative coords
        prediction_list = []
        for tensor_ele in output:
            cls_id = int(tensor_ele[5].item())
            bbox_data = {
                "position": {
                    "minX": tensor_ele[0].item(),
                    "maxX": tensor_ele[2].item(),
                    "minY": tensor_ele[1].item(),
                    "maxY": tensor_ele[3].item(),
                },
                "class_id": cls_id,
                # "box_caption": f"{class_dict[cls_id]}",
                "scores": {
                    "conf": tensor_ele[4].item(),
                },
            }
            prediction_list.append(bbox_data)

        # Construct WandB image obj for logging.
        image = wandb.Image(
            img,
            boxes={"predictions": {"box_data": prediction_list, "class_labels": class_dict},
                   "ground_truth": {"box_data": label_list, "class_labels": class_dict}}
        )
        img_cat_list.append(image)

    wandb.log({wandb_heading: img_cat_list}, commit=False)
    return
