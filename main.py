# Author: Xuechao Zhang
# Date: March 13th, 2022
# Description: 主入口

from src.virtual_cam import *
from src.apriltag_utils import *
from src.xbox_controller import *

if __name__ == "__main__":
    # 创建理想俯视图相机
    IM = (500, 500, -500, -300)
    T = ([0, 0, 0], 2500, 1500, 2500)
    topview = Cam(IM, T, (1049, 1920))

    # 创建第一个普通节点相机
    IM = (1227.678512401081, 1227.856142111345,
                  640, 400)
    T = [[0, 0, 0], -3750, -1300, 1600]
    cam_1 = Cam(IM, T, (800, 1280), distortion=1)

    # 创建第二个普通节点相机
    IM = (1227.678512401081, 1227.856142111345,
                  640, 400)
    T = [[0, 0, 0], -1750, -1450, 3750]
    cam_2 = Cam(IM, T, (800, 1280), distortion=1)

    # 相机列表和当前相机序号
    cameras = [cam_1, cam_2]
    index = 0

    # 设定平面上四个参考点
    reference_points = set_reference_points()
    points_topview = [topview.world_point_to_cam_pixel(point) for point in reference_points[0:4]]

    # 导入背景图
    background = cv2.resize(cv2.imread('./img/grid_with_tags.jpg'), 
                        (topview.img.shape[1], topview.img.shape[0]))

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
            at_print(cam.img, at_detect(cam.img, cam.T44_world_to_cam))

        # 缩放显示
        cv2.imshow('camview_1', cv2.resize(
                cam_1.img, (int(0.5*cam.img.shape[1]), int(0.5*cam.img.shape[0]))))
        cv2.imshow('camview_2', cv2.resize(
                cam_2.img, (int(0.5*cam.img.shape[1]), int(0.5*cam.img.shape[0]))))
        cv2.imshow('topview', cv2.resize(
                topview.img, (int(cam.img.shape[1]), int(cam.img.shape[0]))))

        T_para, command = read_controller(CONTROLLER_TYPE=0)  # 读取控制器
        cameras[index].update_T(*list_add(T_para, cameras[index].T_para))  # 更新外参
        
        if command == -1:  # 解析其他命令
            break
        elif command == 1:
            cameras[index].reset_T()
        elif command == 2:
            index = (index + 1) % len(cameras)
        elif command == 3:
            index = (index - 1) % len(cameras)

    cv2.destroyAllWindows()  # 释放并销毁窗口
