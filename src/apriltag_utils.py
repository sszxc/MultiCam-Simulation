# Author: Xuechao Zhang
# Date: March 13th, 2022
# Description: 调用 AprilTag 检测器进行识别定位

import cv2
import numpy as np
from pupil_apriltags import Detector

class ApriltagDetector:
    def __init__(self, camera_params):
        self.at_detector = Detector(families='tag36h11',
                            nthreads=8,
                            quad_decimate=2.0,
                            quad_sigma=0.0,
                            refine_edges=1,
                            decode_sharpening=0.25,
                            debug=0)
        self.camera_params = camera_params


    def detect(self, frame, T44_world_to_cam):
        '''
        AprilTag 检测
        '''
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tags = self.at_detector.detect(
            frame_gray, estimate_tag_pose=True, 
            camera_params=self.camera_params, tag_size=0.162)
        for tag in tags:
            # 位姿估计
            T44_tag_to_cam = np.c_[np.r_[tag.pose_R, [[0, 0, 0]]], np.r_[tag.pose_t*1000, [[1]]]]  # 拼接成4*4矩阵
            T44_tag_to_world = np.dot(T44_tag_to_cam, T44_world_to_cam)
            # print("tag", tag.tag_id, T44_tag_to_world[0:3, 3])
        return tags

    def draw(self, frame, tags):
        '''
        AprilTag 检测结果标注
        '''
        if tags:
            for tag in tags:
                cv2.circle(frame, tuple(tag.corners[0].astype(
                    int)), 12, (255, 0, 0), 2)  # left-top
                cv2.circle(frame, tuple(tag.corners[1].astype(
                    int)), 12, (255, 0, 0), 2)  # right-top
                cv2.circle(frame, tuple(tag.corners[2].astype(
                    int)), 12, (255, 0, 0), 2)  # right-bottom
                cv2.circle(frame, tuple(tag.corners[3].astype(
                    int)), 12, (255, 0, 0), 2)  # left-bottom
                cv2.putText(frame, str(tag.tag_id), tuple(tag.center.astype(
                    int)), cv2.FONT_HERSHEY_PLAIN, 5.0, (0, 0, 255), 7)
