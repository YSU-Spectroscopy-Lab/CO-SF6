import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
import pickle

def sin_filter(concentration, absor_new, standard_spetrum, wave, file_save,filter_correct_save):
    """加权正弦变换"""

    # 定义成正弦
    def sin_function(x, a, b, c, d):
        return a * np.sin(b * x + c) + d

    # 加权最小二乘法拟合函数
    def weighted_least_squares_fit(x, y, weights):
        popt, pcov = curve_fit(sin_function, x, y, sigma=1 / weights)
        return popt, pcov

    x_stand = np.linspace(0, 2 * np.pi, 50000)  # 横坐标

    # 标准的正弦
    Sin_stand = []
    for x_i in x_stand:
        sin_stand = sin_function(x_i, max(standard_spetrum), 1, 0, 0)
        Sin_stand.append(sin_stand)

    x_stand_all = []
    Sin_stand_temp = np.copy(Sin_stand)
    errors_all = []
    # 将数据填充到曲线上
    for value in standard_spetrum:
        errors = [abs(value - sin_temp) for sin_temp in Sin_stand]
        min_error_index = np.argmin(errors)  # 找出误差值最小的
        Sin_stand[min_error_index] = 1000
        x_stand_part = x_stand[min_error_index]
        x_stand_all.append(x_stand_part)
        errors_all.append(min(errors))

    # 画正弦光谱
    plt.figure(3)
    plt.plot(x_stand, Sin_stand_temp, '-', label='Sin curve', color='b')
    plt.plot(x_stand_all, standard_spetrum, 'o', label='Spectral y_sorted_i', color='r')
    plt.title('Standard concentration')
    plt.xlabel('Data point')
    plt.ylabel('Absorbance')
    plt.title('SIN_reconfiguration')
    plt.legend()
    plt.show()

    # 矫正其他光谱
    OP_all = []
    concentration_all = []
    spectrum_all = []
    for j in range(0, len(absor_new)):
        concentration_0 = concentration[j, :]
        y_sorted_i = np.array(absor_new[j, :])
        sorted_pairs = sorted(zip(x_stand_all, y_sorted_i, wave))
        x_sorted, y_sorted, wave_sorted = zip(*sorted_pairs)
        weights = np.ones_like(x_sorted)
        # 使用加权最小二乘法拟合初始数据
        params, _ = weighted_least_squares_fit(x_sorted, y_sorted, weights)
        # print(params)
        y_0 = sin_function(x_stand, *params)
        y_00 = np.interp(x_sorted, x_stand, y_0)  # 找到新的曲线上  用插值法
        # 计算拟合曲线下的残差
        residuals_0 = y_sorted - y_00

        # 根据残差大小为异常值，赋予较小的权重
        m = 0.001
        variance = np.std(residuals_0)
        # print('方差：', variance)
        threshold = variance * 3  # 设置一个阈值，超过阈值的残差认为是异常值  68-95-99.7  1-2-3
        weights[np.abs(residuals_0) > threshold] = m  # 将超过阈值的残差对应的权重设为0.001

        # 使用加权最小二乘法拟合带有权重的数据
        params_1, _ = weighted_least_squares_fit(x_sorted, y_sorted, weights)
        # print(params_1)
        y_1 = sin_function(x_stand, *params_1)
        y_11 = np.interp(x_sorted, x_stand, y_1)  # 可以完全代替以前的差分光谱数据
        residuals_1 = y_sorted - y_11  # 现在的误差
        colors = np.where(weights > m, 'r', 'k')
        sorted_new = sorted(zip(wave_sorted, y_11))
        wave_new, y_corret = zip(*sorted_new)
        y_original_spectrum = np.array(y_corret)
        OP = sum(abs(y_original_spectrum))
        concentration_all.append(concentration_0)
        spectrum_all.append(y_original_spectrum)
        OP_all.append(OP)

    # 画出原来的光谱
        plt.figure(4)
        plt.plot(wave_new, y_original_spectrum)
        plt.title('SIN_corret')
        plt.draw()  # 刷新
    plt.figure(5)
    plt.plot(concentration_all, OP_all, 'o-', label='Differential OP')
    plt.title('The relationship between filtered spectrum and concentration')
    plt.show()

    df_wave = pd.DataFrame(wave_new)
    df_stand = pd.DataFrame(standard_spetrum)  # 用N2情况下的CO
    df_OP_all = pd.DataFrame(OP_all)
    df_correct = pd.DataFrame(spectrum_all)
    df_concentration = pd.DataFrame(concentration_all)

    # 标准的光谱
    ax = pd.DataFrame(x_stand)
    asst = pd.DataFrame(Sin_stand_temp)
    axs = pd.DataFrame(x_stand_all)
    ass = pd.DataFrame(standard_spetrum)
    "存储的东西都是14个"
    aa = len(spectrum_all)
    if aa > 30:
        data = {'concentration': df_concentration, 'spectrum_all': df_correct, 'wave': df_wave}
        with open(f'data/filter/{file_save}/{filter_correct_save}.pkl', 'wb') as file:
            pickle.dump(data, file)

        with pd.ExcelWriter(f'data/filter/{file_save}/{filter_correct_save}.xlsx') as writer:
            df_correct.to_excel(writer, sheet_name='spectrum_all', index=False, header=False)
            df_concentration.to_excel(writer, sheet_name='concentration', index=False, header=False)
            df_OP_all.to_excel(writer, sheet_name='OP', index=False, header=False)
            df_wave.to_excel(writer, sheet_name='wave', index=False, header=False)
    else:
        data = {'concentration': df_concentration, 'spectrum_all': df_correct, 'wave': df_wave}
        with open(f'data/test/{file_save}/{filter_correct_save}.pkl', 'wb') as file:
            pickle.dump(data, file)

        with pd.ExcelWriter(f'data/test/{file_save}/{filter_correct_save}.xlsx') as writer:
            df_correct.to_excel(writer, sheet_name='spectrum_all', index=False, header=False)
            df_concentration.to_excel(writer, sheet_name='concentration', index=False, header=False)
            df_OP_all.to_excel(writer, sheet_name='OP', index=False, header=False)
            df_wave.to_excel(writer, sheet_name='wave', index=False, header=False)


        with pd.ExcelWriter(f'data/test/{file_save}/Standard_transformation.xlsx') as writer:
            # 标准光谱的数据
            ax.to_excel(writer, sheet_name='x_stand', index=False, header=False)
            asst.to_excel(writer, sheet_name='Sin_stand_temp', index=False, header=False)
            axs.to_excel(writer, sheet_name='x_stand_all', index=False, header=False)
            ass.to_excel(writer, sheet_name='standard_spetrum', index=False, header=False)


    return spectrum_all, concentration_all
