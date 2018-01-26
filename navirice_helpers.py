import navirice_image_pb2
import numpy as np

def navirice_img_set_write_file(session_name, img_set, last_count):
    name = 'DATA/' + session_name + "_" + str(last_count) + ".img_set"
    name = name.replace('\n', '')
    print("Recording to: ", name)
    f = open(name, 'wb')
    f.write(img_set.SerializeToString())
    f.close()


def navirice_image_to_np(img):
    """Should not be called with ir."""
    tp = img.data_type
    divisor = 1
    if(tp == navirice_image_pb2.ProtoImage.FLOAT):
        tp = np.float32
        img.channels = 1
        # IR data comes in short. Dividing the data results in values from 0 to 1
        #divisor = 2**16
        # Depth data to be converted from 0 to 1
        divisor = 4500
    else:
        tp = np.uint8
    rgb_raw = np.frombuffer(img.data, dtype=tp, count=img.width*img.height*img.channels)
    if(img.data_type == navirice_image_pb2.ProtoImage.FLOAT):
        rgb_raw = (rgb_raw) / divisor
    im = rgb_raw.reshape((img.height, img.width, img.channels))
    return im


def navirice_ir_to_np(ir_img):
    """Takes an ir image which has float data, and converts it to a numpy image.
    Does not do any resizing."""
    tp = np.float32
    ir_img.channels = 1
    raw_img= np.frombuffer(ir_img.data, dtype=tp, count=ir_img.width*ir_img.height*ir_img.channels)
    im = raw_img.reshape((ir_img.height, ir_img.width, ir_img.channels))
    return im

def map_depth_and_rgb(rgb_image, depth_image):
    """Takes in rgb_image and depth_image and returns cropped_rgb an cropped_depth."""
    cropped_rgb = rgb_image[:,230:-150]
    cropped_depth = depth_image[30:-30,:]
    return (cropped_rgb, cropped_depth)
