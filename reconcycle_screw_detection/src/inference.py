#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError
import cv2
from ultralytics import YOLO
from colorama import Fore
import numpy as np
import json
import tf
import tf2_ros
from geometry_msgs.msg import TransformStamped
import colorsys

class YOLOv8InferenceNode:
    def __init__(self):
        """
        Initialize the YOLOv8InferenceNode class. This includes setting up the CvBridge for image conversion,
        loading the YOLO model, setting up ROS subscribers and publishers, and initializing camera parameters.
        """
        CAMERA_NAME = "realsense"
        COLOUR_IMG_SUB_TOPIC = f"/{CAMERA_NAME}/color/image_raw"
        DEPTH_IMG_SUB_TOPIC = f"/{CAMERA_NAME}/aligned_depth_to_color/image_raw"
        ANNOTATED_IMG_PUB_TOPIC = f"/{CAMERA_NAME}/color/annotated_screws"
        RESULT_PUB_TOPIC = f"/{CAMERA_NAME}/screw_detections/"
        CAMERA_INFO_TOPIC = f"/{CAMERA_NAME}/color/camera_info"
        self.resultlist = None

        self.bridge = CvBridge()
        # self.yolo_model = YOLO('/root/catkin_ws/src/reconcycle_screw_detection/src/datasets/runs/detect/train/weights/best.pt')
        self.yolo_model = YOLO('datasets/runs/detect/train/weights/best.pt')

        self.image_sub = rospy.Subscriber(COLOUR_IMG_SUB_TOPIC, Image, self.image_callback)
        self.depth_sub = rospy.Subscriber(DEPTH_IMG_SUB_TOPIC, Image, self.depth_callback)
        self.image_pub = rospy.Publisher(ANNOTATED_IMG_PUB_TOPIC, Image, queue_size=10)
        self.result_pub = rospy.Publisher(RESULT_PUB_TOPIC, String, queue_size=10)
        self.tf_pub = tf2_ros.TransformBroadcaster()
        self.camInfo(topic=CAMERA_INFO_TOPIC)

    def image_callback(self, data:Image):
        """
        Callback function for processing the color image from the camera.

        Args:
        -----
            data (Image) : The image message from the camera topic.
        """

        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(data, "rgb8")

        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))
            return

        # Run YOLOv8 inference
        detection_results = self.yolo_model(cv_image)
        print(f"{Fore.MAGENTA}Result type: {type(detection_results)}")
        annotated_image = self.annotate_image(cv_image, detection_results)

        try:
            # Convert OpenCV image to ROS Image message
            ros_image = self.bridge.cv2_to_imgmsg(annotated_image, "bgr8")
            self.image_pub.publish(ros_image)

        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))
            return

    def depth_callback(self, data:Image):
        """
        Callback function for processing the depth image from the camera.

        Args:
        -----
            data (Image): The depth image data from the camera.
        """
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(data, "16UC1")
            
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))
            return

        resultlist_xyz = []

        try:
            for index, result in enumerate(self.resultlist):
                depth_TL = cv_image[result[1]][result[0]]/1000
                depth_BR = cv_image[result[3]][result[2]]/1000
                xyz0 = self.uv_to_XY(result[0], result[1], depth_TL)
                xyz1 = self.uv_to_XY(result[2], result[3], depth_BR)
                center = [(xyz0[0]+xyz1[0])/2, (xyz0[1]+xyz1[1])/2, (xyz0[2] + xyz1[2])/2]

                resultdict = {"box_px":str(result), "box_xyz" : [str(xyz0), str(xyz1)], "box_center" :str(center), "TF_name" : "screw_"+str(index)}
                resultlist_xyz.append(resultdict)

                self.publish_transform(p=center, index=index)

            rosresults = json.dumps(resultlist_xyz)
            self.result_pub.publish(rosresults)

        except Exception as e:
            rospy.logerr("Error: {0}".format(e))
            return
    
    def annotate_image(self, image:cv2.Mat, results:list):
        """
        Process YOLOv8 inference results and draw them on the original image.

        Args:
        ------
            results (str) : YOLOv8 inference results.
            
            image (cv2.Mat) : Original image.

        Returns:
        --------
            image (cv2.Mat) : Annotated image.
        """

        self.resultlist = []
        for result in results:
            for idx, box in enumerate(result.boxes):
                x1, y1, x2, y2 = int(box.xyxy[0][0].item()), int(box.xyxy[0][1].item()), int(box.xyxy[0][2].item()), int(box.xyxy[0][3].item())
                result_px = [x1, y1, x2, y2]
                self.resultlist.append(result_px)
        
                conf = box.conf
                cls = int(box.cls)
                # Draw the bounding box
                step = 255/len(results)*2
                red = int(idx*step)

                colour = (red, 255, 255)
                cv2.rectangle(image, (x1, y1), (x2, y2), colour, 2)
                
                # Draw the label
                label = f"{cls}: {conf}"
                cv2.putText(image, "Screw"+str(idx)+" "+label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)

        return image
    
    def uv_to_XY(self, u:int,v:int, z:int) -> list:
        """
        Convert pixel coordinated (u,v) from realsense camera
        into real world coordinates X,Y,Z 

        Args
        ----
            u (int) : Horizontal coordinate

            v (int) : Vertical coordinate

            z (int) : Depth coordinate

        Returns
        -------
            worldPos (list) : Real world position (in respect to camera)
        """
        
        x = (u - (self.cx)) / self.fx

        y = (v - (self.cy)) / self.fy

        X = (z * x)
        Y = (z * y)
        Z = z

        worldPos = [X, Y, Z]
        return worldPos
    
    def camInfo(self, topic:str):
        """
        Pull camera parameters from the "/camera_info" topic
        
        - Camera matrix 
            - focal lengths: fx, fy
            - optical centres: cx, cy

        Args
        ----
            camera_topic (str) : Specify from which camera we should pull the parameters
        """
        caminfo = rospy.wait_for_message(self.CAMERA_INFO_TOPIC, CameraInfo, timeout=10)
        self.camera_fx = caminfo.K[0]
        self.camera_cx = caminfo.K[2]
        self.camera_fy = caminfo.K[4]
        self.camera_cy = caminfo.K[5]

        self.camMat = np.array([[self.camera_fx, 0.0, self.camera_cx],
                                    [0.0, self.camera_fy, self.camera_cy],
                                    [0.0, 0.0, 1.0]],dtype=np.float32)
        
        self.fx = self.camMat[0, 0]
        self.cx = self.camMat[0, 2]
        self.fy = self.camMat[1, 1]
        self.cy = self.camMat[1, 2]
        
    def publish_transform(self, p:list, index:int):
        """
        Publish a transform for each detected screw.

        Args:
        -----
            p (list): List containing the transform's translation (x, y, z).
            index (int): Index of the detected screw.
        """
        t = TransformStamped()

        # Fill in the header
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = "panda_2/realsense"
        t.child_frame_id = f"screw_{index}"

        # Fill in the transform (translation + rotation)
        t.transform.translation.x = p[0] 
        t.transform.translation.y = p[1]
        t.transform.translation.z = p[2]

        # Quaternion for rotation
        quaternion = tf.transformations.quaternion_from_euler(0, 0, 0)  # roll, pitch, yaw
        t.transform.rotation.x = quaternion[0]
        t.transform.rotation.y = quaternion[1]
        t.transform.rotation.z = quaternion[2]
        t.transform.rotation.w = quaternion[3]

        # Publish the transform
        self.tf_pub.sendTransform(t)



if __name__ == '__main__':
    rospy.init_node('screw_detection', anonymous=False)
    YOLOv8InferenceNode()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down YOLOv8 inference node")