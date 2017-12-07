from navirice_get_image import navirice_get_image
from navirice_head_detect import get_head_from_img
from navirice_helpers import navirice_image_to_np

import cv2
import numpy as np

DEFAULT_HOST = '127.0.0.1'  # The remote host
DEFAULT_PORT = 29000        # The same port as used by the server

def main():
    """Main to test this function. Should not be run for any other reason."""
    last_count = 0
    np.set_printoptions(threshold=np.inf)

    while(1):
        img_set, last_count = navirice_get_image(DEFAULT_HOST, DEFAULT_PORT, last_count)

        if img_set is None:
            continue

        depth_image = navirice_image_to_np(img_set.Depth)
        rgb_image = navirice_image_to_np(img_set.RGB)
        possible_head_data = get_head_from_img(rgb_image)

        working_image = depth_image

        if possible_head_data is None:
            print("No head detected")
            cv2.imshow("working", working_image)
            continue

        (x, y, radius) = possible_head_data

        (scaled_x, scaled_y, scaled_radius) = get_scaled(x, y, radius, working_image)

        resize_scale = 0.3
        cv2.circle(working_image, (scaled_x, scaled_y), scaled_radius, (255, 255, 255), thickness=10, lineType=8, shift=0)
        binary_image = get_labeled_data(depth_image, scaled_x, scaled_y, scaled_radius)
        # Resize if only rgb
        #working_image = cv2.resize(working_image, None, fx=resize_scale, fy=resize_scale, interpolation = cv2.INTER_CUBIC)
        cv2.imshow("working", working_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def get_scaled(x, y, radius, image):
    scaled_x = int(x*image.shape[1])
    scaled_y = int(y*image.shape[0])
    scaled_radius = int(radius*max(image.shape[1], image.shape[0]))
    return (scaled_x, scaled_y, scaled_radius)


def get_labeled_data(depth_image, x, y, radius):
    """Takes in an image and generates labeled images from it.

    Determines the threshold from the center of the crop where the head is."""
    # Todo: make this not a cropped image
    # Todo: account for the rgb to depth mapping
    depth_to_meters = 4.5

    cropped_image = _get_cropped_image(depth_image, x, y, radius)
    modified_image = cropped_image*depth_to_meters

    center_x = int(modified_image.shape[1]/2)
    center_y = int(modified_image.shape[0]/2)
    center_value = modified_image[center_x, center_y]

    # Create threshold of half a meter
    lower_threshold = center_value - center_x
    upper_threshold = center_value + center_y

    # Set everything outside threshold to be white, and within to be black
    white = 255
    black = 0
    modified_image[lower_threshold > modified_image] = white
    modified_image[modified_image > upper_threshold] = white
    modified_image[modified_image != white] = black


def _get_cropped_image(image, x, y, radius):
    width = image.shape[1]
    height = image.shape[0]
    index_crop_left = (x-radius) if (x-radius) > 0 else 0
    index_crop_right = (x+radius) if (x+radius) < width  else width
    index_crop_up = (y-radius) if (y-radius) > 0 else 0
    index_crop_down = (y+radius) if (y+radius) < height else height
    cropped_image = image[index_crop_up:index_crop_down, index_crop_left:index_crop_right]

    # Todo: remove
    cv2.imshow("cropped", cropped_image)

    return cropped_image


if __name__ == "__main__":
    main()
