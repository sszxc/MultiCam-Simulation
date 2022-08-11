# Author: Xuechao Zhang
# Date: March 13th, 2022
# Description: 主入口

import yaml
from src.virtual_cam import *
# from src.apriltag_utils import *
from src.xbox_controller import *

if __name__ == "__main__":
    # 所有相机列表
    all_cameras=[]
    # 从config文件中读取参数 创建虚拟相机
    with open("config/cameras.yaml", "r") as f:
        config = yaml.load(f)
        for camera_name, value in config.items():
            print(camera_name, value)
            all_cameras.append(Cam(camera_name, value))

    # 当前相机序号
    index = 0
    # 理想俯视图相机
    topview = [cam for cam in all_cameras if cam.name == 'topview'][0]
    # 普通节点相机
    cameras = [cam for cam in all_cameras if cam.name != 'topview']

    # 设定平面上四个参考点
    reference_points = set_reference_points()
    points_topview = [topview.world_point_to_cam_pixel(point) for point in reference_points[0:4]]

    # 导入背景图
    background = cv2.resize(cv2.imread('./img/grid_with_tags.jpg'), 
                        (topview.width, topview.height))

    while 1:
        # 手动更新俯视图相机的图像
        topview.img = copy.deepcopy(background)
        
        for cam in cameras:
            # 利用前四个点生成变换矩阵
            cam.update_M(reference_points[0:4], points_topview)

            # 更新本相机图像
            cam.update_img(background)

            # 更新俯视图外框
            cam.cam_frame_project(topview, (140, 144, 32))
        
            # 进行 AprilTag 检测
            # at_print(cam.img, at_detect(cam.img, cam.T44_world_to_cam))

            # 缩放显示
            cv2.imshow(cam.name, cv2.resize(
                    cam.img, (int(0.5*cam.width), int(0.5*cam.height))))
                    
        cv2.imshow('topview', cv2.resize(
                topview.img, (int(cam.width), int(cam.height))))

        T_para, command = read_controller(CONTROLLER_TYPE=0)  # 读取控制器
        cameras[index].update_T(*list_add(T_para, cameras[index].T_para))  # 更新外参
        print(T_para, command)
        
        if command == -1:  # 解析其他命令
            break
        elif command == 1:
            cameras[index].reset_T()
        elif command == 2:
            index = (index + 1) % len(cameras)
        elif command == 3:
            index = (index - 1) % len(cameras)

    cv2.destroyAllWindows()  # 释放并销毁窗口
