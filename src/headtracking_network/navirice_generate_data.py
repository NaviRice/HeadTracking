from navirice_get_image import KinectClient
from navirice_head_detect import get_head_from_img
from navirice_helpers import navirice_image_to_np
from navirice_helpers import map_depth_and_rgb

import cv2
import numpy as np

DEFAULT_HOST = '127.0.0.1'  # The remote host
DEFAULT_PORT = 29000        # The same port as used by the server
#kinect_client = KinectClient(DEFAULT_HOST, DEFAULT_PORT)


def main():
    """Main to test this function. Should not be run for any other reason."""
    # last_count = 0
    # np.set_printoptions(threshold=np.inf)

    # while(1):
    #     img_set, last_count = kinect_client.navirice_get_image()

    #     if(img_set == None or img_set.IR.width == 0 or img_set.Depth.width == 0):
    #         continue

    #     rgb_image = navirice_image_to_np(img_set.RGB)
    #     depth_image = navirice_image_to_np(img_set.Depth)
    #     (rgb_image, depth_image) = map_depth_and_rgb(rgb_image, depth_image)

    #     possible_bitmap = generate_bitmap_label(rgb_image, depth_image)

    #     if possible_bitmap is None:
    #         print("No head detected")
    #         cv2.imshow("depth", depth_image)
    #         continue

    #     binary_image = generate_bitmap_label(rgb_image, depth_image)
    #     cv2.imshow("depth", depth_image)
    #     cv2.imshow("binary", binary_image)

    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    pass


def generate_bitmap_label(rgb_or_ir_image, depth_image):
    """Takes in 2 numpy arrays representing the raw rgb and depth data.

    Uses the first image to do head detection. If rgb image is given,
    assumes it's the same ratio as depth.

    Then uses depth image to generate confidence image. Might have a
    sideeffect of editing depth data.

    returns:
        A bitmap if head is detected, where the head is highlighted with white.
        None if a head is not detected."""

    possible_head_data = get_head_from_img(rgb_or_ir_image)
    if possible_head_data is None:
        bitmap_image = np.zeros(depth_image.shape)
        return bitmap_image
    # (x, y, radius) = possible_head_data # Get the head data as ranges from 0-1
    # (x, y, radius) = get_scaled(x, y, radius, depth_image) # Scale data to depth image
    # bitmap_image = np.zeros(depth_image.shape) # initialize compeletly black label
    # head_location_dimensions = _get_head_location_dimensions(depth_image, x, y, radius)
    # bitmap_image[head_location_dimensions] = 1.0
    # return bitmap_image

    (x, y, radius) = possible_head_data # Get the head data as ranges from 0-1
    (x, y, radius) = get_scaled(x, y, radius, depth_image) # Scale data to depth image
    head_location_dimensions = _get_head_location_dimensions(depth_image, x, y, radius)
    center_value = _get_center_of_head(head_location_dimensions, depth_image)

    head_distance = 1.0 # Create threshold of half a meter
    lower_threshold = center_value - head_distance
    upper_threshold = center_value + head_distance
    # Set everything outside threshold to be black, and within to be white
    black = 0
    depth_to_meters = 4.5
    sub_array_with_head = depth_image[head_location_dimensions] * depth_to_meters
    threshold_indecies = np.logical_or(sub_array_with_head < lower_threshold, upper_threshold < sub_array_with_head)
    sub_array_with_head[threshold_indecies] = black
    sub_array_with_head *= 1.0 
    #sub_array_with_head[sub_array_with_head > 0.0001] = 2.0
    bitmap_image = np.zeros(depth_image.shape) # initialize compeletly black label
    bitmap_image[head_location_dimensions] = sub_array_with_head/depth_to_meters
    bitmap_image[bitmap_image > 0.00001] = 1.0
    return bitmap_image


def get_scaled(x, y, radius, image):
    """Scales x y and raius whose values are from 0 to 1 to int values."""
    scaled_x = int(x*image.shape[1])
    scaled_y = int(y*image.shape[0])
    scaled_radius = int(radius*max(image.shape[1], image.shape[0]))
    return (scaled_x, scaled_y, scaled_radius)


def _get_head_location_dimensions(image, x, y, radius):
    width = image.shape[1]
    height = image.shape[0]
    index_crop_left = (x-radius) if (x-radius) > 0 else 0
    index_crop_right = (x+radius) if (x+radius) < width  else width
    index_crop_up = (y-radius) if (y-radius) > 0 else 0
    index_crop_down = (y+radius) if (y+radius) < height else height
    return  np.index_exp[index_crop_up:index_crop_down, index_crop_left:index_crop_right]


def _get_center_of_head(dimensions, image):
    center_x = dimensions[0].stop-dimensions[0].start
    center_y = dimensions[1].stop-dimensions[1].start
    return image[center_x, center_y]


if __name__ == "__main__":
    main()
