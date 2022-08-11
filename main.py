# Author: Xuechao Zhang
# Date: March 13th, 2022
# Description: 主入口

from src.virtual_cam import *
from src.xbox_controller import *

if __name__ == "__main__":
    env = VirtualCamEnv("config/cameras.yaml", "img/grid_with_tags.jpg")

    while True:
        env.render()
        env.control(*read_controller(CONTROLLER_TYPE=0)) # 读取控制器