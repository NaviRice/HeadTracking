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
    tp = img.data_type
    divisor = 1
    if(tp == navirice_image_pb2.ProtoImage.FLOAT):
        tp = np.float32
        img.channels = 1
        # IR data comes in short. Dividing the data results in values from 0 to 1
        divisor = 2**16
    else:
        tp = np.uint8
    rgb_raw = np.frombuffer(img.data, dtype=tp, count=img.width*img.height*img.channels)
    if(img.data_type == navirice_image_pb2.ProtoImage.FLOAT):
        rgb_raw = (rgb_raw) / divisor
    im = rgb_raw.reshape((img.height, img.width, img.channels))
    return im

