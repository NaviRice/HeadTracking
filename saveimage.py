import OpenEXR
from navirice_get_image import KinectClient
from navirice_helpers import navirice_image_to_np


DEFAULT_HOST= 'navirice'
DEFAULT_PORT=29000



kin = KinectClient(DEFAULT_HOST, DEFAULT_PORT)
kin.navirice_capture_settings(rgb=False, ir=True, depth=True)
last_count=0

img_set, last_count = kin.navirice_get_next_image()

np_depth_image = navirice_image_to_np(img_set.Depth, scale=False)

hdr = OpenEXR.Header(img_set.Depth.width, img_set.Depth.height)
print(hdr)
#hdr['channels'] = {'R': FLOAT(1,1)}
#hdr['channels'] = {'R': hdr['channels']['R']}
print(hdr)

print()
print(img_set.Depth.data_type)
exr = OpenEXR.OutputFile("out.exr", hdr)
exr.writePixels({
'R': np_depth_image,
'G': np_depth_image,
'B': np_depth_image,
})
