# Author: Xuechao Zhang
# Date: Feb 13th, 2022
# Description: 虚拟相机：给定地面纹理和相机位姿，输出图像

import cv2
import numpy as np
import random
import math
import copy
import time
import sys

class Cam:
    def __init__(self, camera_name, parameters):
        '''初始化内外参'''
        self.name = camera_name
        self.expand_for_distortion = 0.15  # 每侧扩张比例
        self.update_IM(*parameters["intrinsic parameters"])
        self.update_T(*parameters["extrinsic parameters"])
        self.T_default = copy.deepcopy(parameters["extrinsic parameters"])  # 存储初始值
        self.height, self.width = parameters["resolution"]
        self.init_view(*parameters["resolution"])
        self.distortion = parameters["distortion"]
        if self.distortion:
            self.init_distortion_para(self.distortion)
        print("camera set!")

    def update_IM(self, fx, fy, cx, cy):
        '''初始化相机内参'''
        self.IM = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]],
            dtype=float
        )

    def EulerToRM(self, theta):
        '''从欧拉角到旋转矩阵'''
        theta = [x / 180.0 * 3.14159265 for x in theta]  # 角度转弧度
        R_x = np.array([[1,                  0,                   0],
                        [0, math.cos(theta[0]), -math.sin(theta[0])],
                        [0,  math.sin(theta[0]), math.cos(theta[0])]])
        R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                        [0,                  1,                  0],
                        [-math.sin(theta[1]), 0, math.cos(theta[1])]])
        R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                        [math.sin(theta[2]), math.cos(theta[2]),  0],
                        [0,                  0,                   1]])
        R = np.dot(R_y, np.dot(R_x, R_z))
        return R

    def update_T(self, euler, dx, dy, dz):
        '''
        更新外参 即相机与世界之间的转换矩阵
        更新相机中轴线与地面交点
        '''
        self.T_para = [euler, dx, dy, dz]
        self.dz = dz
        RM = self.EulerToRM(euler)
        T_43 = np.r_[RM, [[0, 0, 0]]]
        T_rotate = np.c_[T_43, [0, 0, 0, 1]]
        T_trans = np.array([
            [1, 0, 0, dx],
            [0, 1, 0, dy],
            [0, 0, 1, dz],
            [0, 0, 0, 1]],
            dtype=float
        )
        self.T44_cam_to_world = np.dot(T_rotate, T_trans)
        self.T44_world_to_cam = np.linalg.inv(self.T44_cam_to_world)
        self.update_central_axis_cross_ground()  # 更新光轴交点

    def update_central_axis_cross_ground(self):
        '''计算光轴与地面交点'''
        euler, dx, dy, dz = self.T_para
        # 竖直向下的向量经过RM矩阵旋转得到法向
        direction = np.dot(
            np.array([0, 0, 1], dtype=np.float), self.EulerToRM(euler))  # 方向向量
        focus_x = dx - dz/direction[2]*direction[0]
        focus_y = dy - dz/direction[2]*direction[1]
        self.central_axis_cross_ground = np.array([[-focus_x], [-focus_y], [0]], dtype=float)

    def reset_T(self):
        '''外参归位'''
        self.update_T(*self.T_default)

    def update_M(self, points_3D, points_topview):
        '''
        用几组点对计算相机到全局的图片变换矩阵
        '''
        pixels = [self.world_point_to_cam_pixel(point) for point in points_3D]
        # 为畸变模拟服务 目标位置平移
        if self.distortion:
            pixels_expand = []
            for i in pixels:
                pixels_expand.append((i[0] + self.expand_for_distortion*self.height,
                                i[1] + self.expand_for_distortion*self.width))
            self.M33_global_to_local_expand = cv2.getPerspectiveTransform(np.float32(points_topview), 
                                                                    np.float32(pixels_expand))
        # 非畸变情况
        self.M33_global_to_local = cv2.getPerspectiveTransform(np.float32(points_topview), 
                                                                np.float32(pixels))
        # 注意这里畸变情况下也用标准的 local_to_global 仅在投影到俯视图中会调用
        self.M33_local_to_global = np.linalg.inv(self.M33_global_to_local)

    def init_view(self, height=600, width=1000, color=(255, 255, 255)):
        '''
        新建图像，并设置分辨率、背景色
        '''
        graph = np.zeros((height, width, 3), np.uint8)
        graph[:, :, 0] += color[0]
        graph[:, :, 1] += color[1]
        graph[:, :, 2] += color[2]
        self.img = graph

    def world_point_to_cam_point(self, point):
        '''
        从世界坐标点到相机坐标点
        补上第四维的齐次 1, 矩阵乘法, 去掉齐次 1
        '''
        point_in_cam = np.dot(self.T44_cam_to_world, np.r_[point, [[1]]])
        return np.delete(point_in_cam, 3, 0)

    def world_point_to_cam_pixel(self, point):
        '''
        从世界坐标点到像素坐标点 → “拍照”
        '''
        point_in_cam = self.world_point_to_cam_point(point)
        dot = np.dot(self.IM, point_in_cam)
        dot /= dot[2]  # 归一化
        pixel = tuple(dot.astype(np.int).T.tolist()[0][0:2])
        return pixel

    def cam_frame_project(self, topview, color):
        '''
        根据俯视相机参数 把相机视野轮廓投影到俯视图
        '''
        # 节点相机的角点
        corners = []
        if self.distortion:  # 畸变情况下 由于画面变化 轮廓角点也要调整
            scale = self.expand_for_distortion
            corners.append(np.array([scale*self.width, scale*self.height, 1], dtype=np.float32))
            corners.append(np.array([(scale+1)*self.width, scale*self.height, 1], dtype=np.float32))
            corners.append(np.array([scale*self.width, (scale+1)*self.height, 1], dtype=np.float32))
            corners.append(np.array([(scale+1)*self.width, (scale+1)*self.height, 1], dtype=np.float32))
        else:
            corners.append(np.array([0, 0, 1], dtype=np.float32))
            corners.append(np.array([self.width, 0, 1], dtype=np.float32))
            corners.append(np.array([0, self.height, 1], dtype=np.float32))
            corners.append(np.array([self.width, self.height, 1], dtype=np.float32))
        # 从相机角点到投影图
        projected_corners = []
        for corner in corners:
            projected_corner = np.dot(self.M33_local_to_global, corner.T)
            projected_corner /= projected_corner[2]  # 归一化
            projected_corners.append(tuple(projected_corner.astype(
                np.int).T.tolist()[0:2]))
            cv2.circle(topview.img, tuple(projected_corner.astype(
                np.int).T.tolist()[0:2]), 10, color, -1)
        cv2.line(topview.img, projected_corners[0], projected_corners[1], color,5)
        cv2.line(topview.img, projected_corners[1], projected_corners[3], color,5)
        cv2.line(topview.img, projected_corners[2], projected_corners[3], color,5)
        cv2.line(topview.img, projected_corners[2], projected_corners[0], color,5)
        
        # 标记相机视角中心
        focus_center = topview.world_point_to_cam_pixel(self.central_axis_cross_ground)
        color_for_grid = (130, 57, 68)
        color_for_testbed = (0, 255, 255)
        cv2.circle(topview.img, focus_center, 10, color_for_grid, -1)
        
        # 标记相机固定中心的投影
        camera_center_projected = topview.world_point_to_cam_pixel(np.array([[-self.T_para[1]], [-self.T_para[2]], [0]]))
        cv2.line(topview.img, projected_corners[0], camera_center_projected, color,2)
        cv2.line(topview.img, projected_corners[1], camera_center_projected, color,2)
        cv2.line(topview.img, projected_corners[2], camera_center_projected, color,2)
        cv2.line(topview.img, projected_corners[3], camera_center_projected, color,2)
        color_for_grid = (69, 214, 144)
        cv2.circle(topview.img, camera_center_projected, 10, color_for_grid, -1)

    def init_distortion_para(self, distortion_parameters):
        '''一组正向模拟畸变系数'''
        k1, k2, k3, k4, k5, k6, p1, p2 = distortion_parameters
        k = self.IM

        scale = 2*self.expand_for_distortion+1
        k[0,2]*=scale
        k[1,2]*=scale
        d = np.array([
            k1, k2, p1, p2, k3, k4, k5, k6
        ])
        self.distortion_mapx, self.distortion_mapy = cv2.initUndistortRectifyMap(
            k, d, None, k, (int(self.width*scale), int(self.height*scale)), 5)

    def update_img(self, background, DEBUG = 0):
        '''更新图像 同时实现可选的畸变'''
        if self.distortion:
            # 扩展画布
            scale = self.expand_for_distortion*2+1
            self.img_expand = cv2.warpPerspective(
                background, self.M33_global_to_local,
                (int(self.width*scale), int(self.height*scale)),
                borderValue=(255, 255, 255))
            # 正向模拟畸变
            self.img_expand = cv2.remap(self.img_expand, 
                self.distortion_mapx, self.distortion_mapy, 
                cv2.INTER_LINEAR)
            # 导出没有扩张视野的图像
            if DEBUG:
                # 调试看全貌
                self.img = cv2.resize(self.img_expand, (self.img.shape[1], self.img.shape[0]))
            else:
                y_0=int(self.expand_for_distortion*self.height)
                y_1=int((1+self.expand_for_distortion)*self.height)
                x_0=int(self.expand_for_distortion*self.width)
                x_1=int((1+self.expand_for_distortion)*self.width)
                self.img = self.img_expand[y_0:y_1, x_0:x_1]
        else:
            self.img = cv2.warpPerspective(
                background, self.M33_global_to_local, 
                (self.width, self.height), 
                borderValue=(255, 255, 255))

def list_add(a, b):
    '''一种将list各个元素相加的方法'''
    c = []
    for i, j in zip(a,b):
        if not isinstance(i,list):  # 如果不是list那直接相加
            c.append(i+j)
        else:  # 否则递归
            c.append(list_add(i,j))
    return c

def randomColor():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

def set_reference_points():    
    '''在地面上设置四个定位空间点'''
    points = [np.array([[0], [0], [0]], dtype=float)]*4
    points[0] = np.array([[0], [0], [0]], dtype=float)
    points[1] = np.array([[0], [3000], [0]], dtype=float)
    points[2] = np.array([[5000], [3000], [0]], dtype=float)
    points[3] = np.array([[5000], [0], [0]], dtype=float)
    return points

def undistort(frame):
    '''
    畸变矫正
    '''
    if 'mapx' not in globals():
        global mapx, mapy
        # 一组去畸变参数
        fx = 827.678512401081
        cx = 640
        fy = 827.856142111345
        cy = 400
        k1, k2, p1, p2, k3 = -0.335814019871572, 0.101431758719313, 0.0, 0.0, 0.0

        # 相机坐标系到像素坐标系的转换矩阵
        k = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
        # 畸变系数
        d = np.array([
            k1, k2, p1, p2, k3
        ])
        h, w = frame.shape[:2]
        mapx, mapy = cv2.initUndistortRectifyMap(k, d, None, k, (w, h), 5)
    return cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)

def save_image(img, rename_by_time, path="/img/"):
    '''
    True: 按照时间命名 False: 按照序号命名
    '''
    filename = sys.path[0] + path
    if rename_by_time:
        filename += time.strftime('%H%M%S')
    else:
        if 'index' not in globals():
            global index
            index = 1  # 保存图片起始索引
        else:
            index += 1
        filename += str(index)
    filename += ".jpg"
    # cv2.imwrite("./img/" + filename, img) # 非中文路径保存图片
    cv2.imencode('.jpg', img)[1].tofile(filename)  # 中文路径保存图片
    print("save img successfuly!")
    print(filename)

if __name__ == "__main__":
    IM = (500, 500, -500, -300)
    T = ([0, 0, 0], 2500, 1500, 2500)
    topview = Cam(IM, T, (1049, 1920))