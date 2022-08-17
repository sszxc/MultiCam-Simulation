import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def distortion_func_2para(x, k1, k2):
    return 1+k1*x**2+k2*x**4

def distortion_func_3para(x, k1, k2, k3):
    return 1+k1*x**2+k2*x**4+k3*x**6

def distortion_func_6para(x, k1, k2, k3, k4, k5, k6):
    return x*(1+k1*x**2+k2*x**4+k3*x**6)/(1+k4*x**2+k5*x**4+k6*x**6)

def generate_data(func, para):
    x = np.linspace(0,1.5)
    return x, x*func(x, *para)

def para_fit(x_real_extanded, y_real_extanded, func):
    x_real=[]
    y_real=[]
    for i in range(len(x_real_extanded)):  # 截取[0,1]区间
        if y_real_extanded[i] < 1.0:
            x_real.append(x_real_extanded[i])
            y_real.append(y_real_extanded[i])
    x_real = np.array(x_real)
    y_real = np.array(y_real)

    popt, pcov = curve_fit(func, y_real, x_real, maxfev=5000)  # 拟合
    return popt, pcov

if __name__ == "__main__":
    k1 = -0.335814019871572
    k2 = 0.101431758719313
    
    x_real_extanded, y_real_extanded = generate_data(distortion_func_2para, (k1, k2))  # 真实数据

    plt.plot(x_real_extanded, x_real_extanded, 'g-', label='origin')
    plt.plot(x_real_extanded, y_real_extanded, 'b-', label='distort')
    plt.plot(y_real_extanded, x_real_extanded, 'b-', label='target')

    popt, pcov = para_fit(x_real_extanded, y_real_extanded, distortion_func_6para)
    print(popt)
    plt.plot(x_real_extanded, distortion_func_6para(x_real_extanded, *popt), 'r*',
            label='fit: k1=%5.3f, k2=%5.3f, k3=%5.3f, k4=%5.3f, k5=%5.3f, k6=%5.3f' % tuple(popt))
    

    # plt.xlim(0, 1)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()