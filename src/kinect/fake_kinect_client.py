from time import sleep
import settings

import navirice_image_pb2


class FakeKinectClient:
    def __init__(self, host, port):
        """Takes in host and port to be similar to KinectClient."""
        self.last_count = 0

    def navirice_get_image(self, mode="All"):
        """Returns an image from a folder based on mode.

        Default mode is "irdance"
        Current Available Modes (get data from DATA folder):
            - irdance
            - default
        Upcomming Modes:
            - All
            - head
            - nohead"""
        img_set = None
        while img_set is None:
            try:
                file_location = (settings.DATA_DIR + mode
                    + "_" + str(self.last_count)
                    + ".img_set")
                f = open(file_location, "rb")
                data = f.read()
                img_set = navirice_image_pb2.ProtoImageSet()
                img_set.ParseFromString(data)
            except FileNotFoundError:
                # File not found, so try another count
                if self.last_count >= 10000:
                    print("Exceeded 10000 images to check, looping")
                    return None, self.last_count
            self.last_count += 1
        sleep(0.03)
        return img_set, self.last_count

    def navirice_capture_settings(self, rgb, ir, depth):
        """Takes in rgb, ir, and depth to be similar to KinectClient."""
        pass

    def navirice_get_next_image(self, mode="default"):
        potential_image = self.navirice_get_image(mode)
        if potential_image == None:
            self.last_count = 1
            return self.navirice_get_image()
        else:
            return potential_image

