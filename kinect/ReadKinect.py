import freenect
import cv2
import numpy as np
from TrackObject.track_object import track_object

class read_image():
    def __init__(self):
        self.array,_=freenect.sync_get_video()

    def get_video(self):
        array = cv2.cvtColor(self.array,cv2.COLOR_RGB2BGR)
        return array

    def track_object(self):
        while True:
            frame=self.get_video()
            k = cv2.waitKey(5) & 0xFF
            if k == 27:
                break
            track_object.get_image(frame)
        cv2.destroyAllWindows()
