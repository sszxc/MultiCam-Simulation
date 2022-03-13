# Author: Xuechao Zhang
# Date: March 22th, 2021
# Description: 获取Xbox手柄的输入作为调试
#               适用于 Xbox One 手柄 XPS15 电脑
#               DELL7070 电脑上仍然有获取按键 bug
#               可用蓝牙/有线连接
#               先在控制面板→设备和打印机找一下Controller设备

import pygame
import time
import cv2

def joyinit():
    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count():
        joystick = pygame.joystick.Joystick(0)
        joystick.init()
        # axes = joystick.get_numaxes() # 6个
        # buttons = joystick.get_numbuttons()  # 16个 只知道十个有用
        # hats = joystick.get_numhats()  # 1个
        return joystick
    return 0

def joystick_input(joystick, threshold=0.1):
    # for event in pygame.event.get():  # User did something
    #     if event.type == pygame.QUIT:  # If user clicked close
    #         done = True  # Flag that we are done so we exit this loop

    pygame.event.get()

    axis = [joystick.get_axis(i) for i in range(joystick.get_numaxes())] # 摇杆
    button = [joystick.get_button(i) for i in range(joystick.get_numbuttons())]  # 按键
    hat = [joystick.get_hat(i) for i in range(joystick.get_numhats())]  # 十字键

    return filter(axis, threshold), button, hat

def filter(axis, threshold):
    '''去除小量漂移'''
    return [value if abs(value)>threshold else 0 for value in axis]


def read_controller(CONTROLLER_TYPE=1):
    '''
    CONTROLLER_TYPE: 不同电脑对手柄的配置不同 0:XPS15 1:DELL7070
    command: -1退出；1复位；23相机切换
    手柄: 左摇杆水平位移 右摇杆角度 LT&RT高度 A退出 B复位 LR切换相机
    键盘: WASD平移 ZX高度 UJIKOL旋转 0复位 Q退出
    '''
    T_para = [[0, 0, 0], 0, 0, 0]
    command = 0
    if 'joystick' not in globals():
        # 初始化手柄控制
        global joystick
        joystick = joyinit()
    if joystick:
        if CONTROLLER_TYPE == 0:
            axis, button, hat = joystick_input(joystick)
            T_para[1] += axis[0] * -50
            T_para[2] += axis[1] * -50
            T_para[3] += hat[0][1] * -30
            T_para[0][0] = axis[3] * 1
            T_para[0][1] = axis[2] * -1
            T_para[0][2] += (axis[4] - axis[5]) * 0.5
            if button[0] == 1:
                command = -1
            elif button[1] == 1:
                command = 1
            elif button[4] == 1:
                while button[4] == 1:  # 等待松开
                    _, button, _ = joystick_input(joystick)
                command = 2
            elif button[5] == 1:
                while button[5] == 1:
                    _, button, _ = joystick_input(joystick)
                command = 3

        elif CONTROLLER_TYPE == 1:
            axis, button, hat = joystick_input(joystick)
            T_para[1] += axis[0] * -30
            T_para[2] += axis[1] * -30
            T_para[3] += axis[2] * 50
            T_para[0][1] = axis[3] * -50
            T_para[0][0] = axis[4] * 50
            if button[0] == 1:
                command = -1
            elif button[1] == 1:
                command = 1

    else:
        k = cv2.waitKey(0) & 0xFF
        if k == ord('q') or k == ord('Q'):
            command = -1
        elif k == ord('A'):
            T_para[1] += 250
        elif k == ord('D'):
            T_para[1] -= 250
        elif k == ord('W'):
            T_para[2] += 250
        elif k == ord('S'):
            T_para[2] -= 250
        elif k == ord('Z'):
            T_para[3] += 250
        elif k == ord('X'):
            T_para[3] -= 250
        elif k == ord('U'):
            T_para[0][0] += 10
        elif k == ord('J'):
            T_para[0][0] -= 10
        elif k == ord('I'):
            T_para[0][1] += 10
        elif k == ord('K'):
            T_para[0][1] -= 10
        elif k == ord('O'):
            T_para[0][2] += 10
        elif k == ord('L'):
            T_para[0][2] -= 10
        elif k == ord('1'):
            command = 1
        elif k == ord('2'):
            command = 2
        elif k == ord('3'):
            command = 3

    return T_para, command

if __name__ == "__main__":
    joystick = joyinit()

    size = [500, 700]
    screen = pygame.display.set_mode(size)
    # pygame.display.set_caption("My Game")

    if joystick:
        while (1):
            axis, button, hat = joystick_input(joystick)
            print(axis)
            print(button)
            print(hat)
            time.sleep(0.1)
    pygame.quit()
