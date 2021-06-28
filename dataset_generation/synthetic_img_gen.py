import numpy as np
import cv2
import random
import argparse
from pathlib import Path

def overlay_obj_on_canvas(source_img, output_img, source_img_class_id):
    # Randomly geneate Y and X coordinate values where the source_image needs to be pasted
    height = output_img.shape[0]
    width = output_img.shape[1]

    # extract source image height and width
    source_img_height = source_img.shape[0]
    source_img_width = source_img.shape[1]

    #generate random coordinates
    rnd_y_coord = random.randint(0, height-source_img_height)
    rnd_x_coord = random.randint(0, width-source_img_width)

    # Paste the source image on a random coordinate
    start_pos = (rnd_y_coord,  rnd_x_coord)
    end_pos = (rnd_y_coord + source_img_height,  rnd_x_coord + source_img_width)
    output_img[start_pos[0]:end_pos[0], start_pos[1]:end_pos[1], :] = source_img

    # Store the coordinates of the bbox (class_id, xmin, ymin, xmax, ymax)
    bbox_coordinate = np.array([[source_img_class_id, start_pos[1], start_pos[0], end_pos[1], end_pos[0]]], dtype=np.int32) # (1,5) shape

    return bbox_coordinate , output_img


if __name__ == "__main__":
    # Dictionary of source image path
    source_imgs = {
        0: "/Users/samiksha/master_thesis/images_from_drive_nippon/my_dataset/car_small.png",  # car small
        1: "/Users/samiksha/master_thesis/images_from_drive_nippon/my_dataset/car_small_g.png",  # car small green
        2: "/Users/samiksha/master_thesis/images_from_drive_nippon/my_dataset/car_big.png", # car big
        3: "/Users/samiksha/master_thesis/images_from_drive_nippon/my_dataset/car_big_g.png",  # car big green
        4: "/Users/samiksha/master_thesis/images_from_drive_nippon/my_dataset/car_big_r.png",  # car big red
        5: "/Users/samiksha/master_thesis/images_from_drive_nippon/my_dataset/one_dash.png" ,#one dash
        6: "/Users/samiksha/master_thesis/images_from_drive_nippon/my_dataset/two_dash.png",  # two dash
        7: "/Users/samiksha/master_thesis/images_from_drive_nippon/my_dataset/three_dash.png" , #three dash
        8: "/Users/samiksha/master_thesis/images_from_drive_nippon/my_dataset/four_dash.png" , #four dash
        9: "/Users/samiksha/master_thesis/images_from_drive_nippon/my_dataset/five_dash.png",  # five dash
        10: "/Users/samiksha/master_thesis/images_from_drive_nippon/my_dataset/Dotted_lines_LR.png", # dotted lanes white
        11: "/Users/samiksha/master_thesis/images_from_drive_nippon/my_dataset/dotted_lanes_g.png" , # dotted lanes green
        12: "/Users/samiksha/master_thesis/images_from_drive_nippon/my_dataset/dotted_lanes_LR_r.png" , # dotted lanes red
        13: "/Users/samiksha/master_thesis/images_from_drive_nippon/my_dataset/brackets_LR.png",  # brackets LR
        14: "/Users/samiksha/master_thesis/images_from_drive_nippon/my_dataset/brackets_LR_g.png",  # brackets LR green
        15: "/Users/samiksha/master_thesis/images_from_drive_nippon/my_dataset/triangles_LR.png",  # triangles LR
        16: "/Users/samiksha/master_thesis/images_from_drive_nippon/my_dataset/triangles_LR_g.png",  # triangles LR green
        17: "/Users/samiksha/master_thesis/images_from_drive_nippon/my_dataset/triangles_LR_r.png",  # triangles LR red
    }

    # accept values of number of images to generate and directory in which the images need to be stored
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_output", type=Path, default="output",
                        help="Directory to store generated img and txt file")
    parser.add_argument("--num_imgs", type=Path, default=2000,
                        help="Number of images to generate")
    args = parser.parse_args()

    dir_output = args.dir_output
    # create directory output
    dir_output.mkdir(parents=True, exist_ok=True)
    dir_images = dir_output / "images"
    dir_annotations = dir_output / "annotations"
    dir_images.mkdir(parents=True, exist_ok=True)
    dir_annotations.mkdir(parents=True, exist_ok=True)

    num_of_output_imgs = args.num_imgs
    for output_img_idx in range(num_of_output_imgs):

        # decide how many source image to pick
        max_source_images = min(len(source_imgs), 4)
        number_of_source_images_to_use = random.randint(1, max_source_images)

        # Create a black image canvas of given dimension
        output_img = np.zeros((256, 256, 3), dtype=np.uint8)

        bbox_coordinates = []
        for image_idx in range(number_of_source_images_to_use):

            # randomly select 1 source image from the dictionary
            source_img_dict_list = list(source_imgs.items())
            random_source_img = random.choice(source_img_dict_list) #randomly chose 1 source image from the list of items from dict
            random_source_img_path = random_source_img[1]
            random_source_img_class_id = random_source_img[0]

            # Read the image
            source_img = cv2.imread(random_source_img_path)
            #print("source_img", source_img.shape)

            # Create bbox coordinate list
            source_img_coordinates , output_img = overlay_obj_on_canvas(source_img, output_img, random_source_img_class_id)
            bbox_coordinates.append(source_img_coordinates)

        # Convert the list of bbox coordinates into numpy array to save it into a text
        bbox_numpy = np.concatenate(bbox_coordinates, axis=0)
        f_annotation = dir_annotations / f'{output_img_idx:04d}.txt'
        np.savetxt(str(f_annotation), bbox_numpy, delimiter=',', fmt='%d')

        # Save the output image in a file
        f_images = dir_images / f'{output_img_idx:04d}.png'
        cv2.imwrite(str(f_images), output_img)

        # Visualize the bboxes
        dir_debug = dir_output / "debug"
        dir_debug.mkdir(parents=True, exist_ok=True)
        f_debug_img = dir_debug / f"{output_img_idx:04d}.png"
        output_img_debug = output_img.copy()
        for bbox in bbox_numpy:
            start_point = (bbox[1], bbox[2])
            end_point = (bbox[3], bbox[4])
            color = (255, 0, 0)
            thickness = 2
            output_img_debug = cv2.rectangle(output_img_debug, start_point, end_point, color, thickness)
        cv2.imwrite(str(f_debug_img), output_img_debug)

