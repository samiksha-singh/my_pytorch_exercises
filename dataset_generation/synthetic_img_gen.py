import numpy as np
import cv2
import random


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
        0: "/Users/samiksha/master_thesis/images_from_drive_nippon/my_dataset/car_edit.png",  # car
        1: "/Users/samiksha/master_thesis/images_from_drive_nippon/my_dataset/1521_edit.png",  # 5 dashes
        2: "/Users/samiksha/master_thesis/images_from_drive_nippon/my_dataset/430_edit.png" # dotted lanes
    }

    num_of_output_imgs = 5
    for output_img_idx in range(num_of_output_imgs):

        # decide how many source image to pick
        max_source_images = min(len(source_imgs), 5)
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
        np.savetxt(f'{output_img_idx:04d}.txt', bbox_numpy, delimiter=',', fmt='%d')

        # Save the output image in a file
        cv2.imwrite(f'{output_img_idx:04d}.png', output_img)
