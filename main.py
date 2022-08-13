# Author: Xuechao Zhang
# Date: March 13th, 2022
# Description: 主入口

from src.virtual_cam import *
from src.xbox_controller import *

if __name__ == "__main__":
    env = VirtualCamEnv("config/cameras.yaml", 
                        "img/grid_with_tags.jpg", 
                        AprilTag_detection=True)

    while True:
        env.control(*read_controller(CONTROLLER_TYPE=0)) # 读取控制器
        imgs = env.render() # 渲染图像
        
        # 缩放显示
        for (img, name) in imgs:
            cv2.imshow(name, cv2.resize(img, (int(0.5*img.shape[1]), int(0.5*img.shape[0]))))