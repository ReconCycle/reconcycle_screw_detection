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
        self.bridge = CvBridge()
        self.yolo_model = YOLO('/root/catkin_ws/src/reconcycle_screw_detection/src/datasets/runs/detect/train/weights/best.pt')
        self.image_sub = rospy.Subscriber("/realsense/color/image_raw", Image, self.image_callback)
        self.depth_sub = rospy.Subscriber("/realsense/aligned_depth_to_color/image_raw", Image, self.depth_callback)
        self.result_pub = rospy.Publisher("/screwdetections/yolov8_screws", String, queue_size=10)
        self.image_pub = rospy.Publisher("/realsense/yolov8_screws", Image, queue_size=10)
        self.tf_pub = tf2_ros.TransformBroadcaster()
        self.camInfo()
        self.resultlist = None

    def image_callback(self, data):
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))
            return

        # Run YOLOv8 inference
        results = self.yolo_model(cv_image)
        
        annotated_image = self.draw_results(cv_image, results)
        

        try:
            # Convert OpenCV image to ROS Image message
            ros_image = self.bridge.cv2_to_imgmsg(annotated_image, "bgr8")
            self.image_pub.publish(ros_image)
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))
            return

    def depth_callback(self, data):
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(data, "16UC1")
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))
            return
        resultlist_xyz = []
        try:
            for index, result in enumerate(self.resultlist):
                depth0 = cv_image[result[1]][result[0]]/1000
                depth1 = cv_image[result[3]][result[2]]/1000
                xyz0 = self.uv_to_XY(result[0], result[1], depth0)
                xyz1 = self.uv_to_XY(result[2], result[3], depth1)

                resultxy = {"box_px":str(result), "box_xyz" : [str(xyz0), str(xyz1)], "TF_name" : "screw_"+str(index)}
                center = [(xyz0[0]+xyz1[0])/2, (xyz0[1]+xyz1[1])/2, (xyz0[2] + xyz1[2])/2]
                self.publish_transform(p=center, index=index)
                resultlist_xyz.append(resultxy)

            rosresults = json.dumps(resultlist_xyz)
            # Convert OpenCV image to ROS Image message
            self.result_pub.publish(rosresults)
        except Exception as e:
            rospy.logerr("Error: {0}".format(e))
            return

    def process_results(self, results):
        # Convert results to a string (or any other format you need)
        result_str = ""
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0][0].item(), box.xyxy[0][1].item(), box.xyxy[0][2].item(), box.xyxy[0][3].item()
                result_str += f"BBox: ({x1}, {y1}, {x2}, {y2})\n"
        return result_str
    
    def draw_results(self, image, results):
        # Draw bounding boxes and labels on the image
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
            u(int) : Horizontal coordinate

            v(int) : Vertical coordinate

            z(int) : Depth coordinate

        Returns
        -------
            worldPos(list) : Real world position (in respect to camera)
        """
        
        #x = (u - (496.91)) / 635.7753
        x = (u - (self.cx)) / self.fx

        #y = (v - (489.182)) / 355.61024
        y = (v - (self.cy)) / self.fy

        X = (z * x)
        Y = (z * y)
        Z = z

        worldPos = [X, Y, Z]
        return worldPos
    
    def camInfo(self):
        """
        This function pulls camera parameters from the "/camera_info" topic
        - Camera matrix 
            - focal lengths: fx, fy
            - optical centres: cx, cy
        - Distortion coefficients

        Args
        ----
            camera_topic (str) : Specify from which camera we should pull the parameters
        """
        caminfo = rospy.wait_for_message('/realsense/color/camera_info', CameraInfo, timeout=10)
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
        t = TransformStamped()

        # Fill in the header
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = "panda_2/realsense"
        t.child_frame_id = "screw_"+str(index)

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