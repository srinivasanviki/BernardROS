
#!/usr/bin/env python
import rospy
from std_msgs.msg import Empty
from std_msgs.msg import String
from rosgraph_msgs.msg import Log

from sensor_msgs.msg import Image


import numpy as np
import matplotlib.pyplot as plt
import cv2
import cv_bridge
from cv_bridge import CvBridge, CvBridgeError

obj_found = False

correct_location = True

obj_color = 0

xb = 0
yb = 0

class track_object():
    def __init__(self):
        self.bridge= CvBridge()

    def get_image(self,frame):

        cv_image = self.bridge.imgmsg_to_cv2(frame, "bgr8")
        height,width,depth=cv_image.shape
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        thresholded = 0
        obj_color = 0

        if obj_color == 0:
            greenLower = (29, 86, 6)
            greenUpper = (64, 255, 255)
            thresholded = cv2.inRange(hsv, greenLower,greenUpper)

            thresholded = cv2.erode(thresholded, None, iterations=2)
            thresholded = cv2.dilate(thresholded, None, iterations=2)

            contours = cv2.findContours(thresholded.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)[-2]

            cv2.drawContours(cv_image, contours, -1, (0,255,0), 3)
            center = None

            numobj = len(contours)

            if numobj > 0:
                moms = cv2.moments(contours[0])
                if moms['m00']>500:
                    cx = int(moms['m10']/moms['m00'])
                    cy = int(moms['m01']/moms['m00'])



                    xb = (cy - (height/2))*.0023*.433 + .712 + .02
                    yb = (cx - (width/2))*.0023*.433 + .316  - .02
                obj_found = True
            else:
                obj_color = 1 #No green objects were found so switch to red
                # Analyze image for red objects
                low_h  = 0
                high_h = 3
                low_s  = 130
                high_s = 190
                low_v  = 80
                high_v = 250

                thresholded = cv2.inRange(hsv, np.array([low_h, low_s, low_v]), np.array([high_h, high_s, high_v]))

                #Morphological opening (remove small objects from the foreground)
                thresholded = cv2.erode(thresholded, np.ones((2,2), np.uint8), iterations=1)
                thresholded = cv2.dilate(thresholded, np.ones((2,2), np.uint8), iterations=1)

                #Morphological closing (fill small holes in the foreground)
                thresholded = cv2.dilate(thresholded, np.ones((2,2), np.uint8), iterations=1)
                thresholded = cv2.erode(thresholded, np.ones((2,2), np.uint8), iterations=1)

                ret,thresh = cv2.threshold(thresholded,157,255,0)

                contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

                cv2.drawContours(cv_image, contours, -1, (0,255,0), 3)

                numobj = len(contours) # number of objects found in current frame
                #print 'Number of objects found in the current frame: ' , numobj

                if numobj > 0:
                    moms = cv2.moments(contours[0])
                    if moms['m00']>500:
                        cx = int(moms['m10']/moms['m00'])
                        cy = int(moms['m01']/moms['m00'])


                        xb = (cy - (height/2))*.0023*.433 + .712 + .02
                        yb = (cx - (width/2))*.0023*.433 + .316  - .02

                    obj_found = True
        else:
            print "Couldn't find any green or blue objects."


        cv2.imshow("Original", cv_image)

        cv2.imshow("Thresholded", thresholded)
        cv2.waitKey(3)

    def get_obj_location(self):
        global xb, yb, obj_color, correct_location, obj_found

        while xb == 0 and yb == 0:
            rospy.sleep(1)
        return xb,yb,obj_found,correct_location










if __name__ == '__main__':
    rospy.init_node('kinect', anonymous=True)
    topic="/kinect2/sd/image_depth"
    rospy.Subscriber(topic,Image, get_image)

    rospy.spin()