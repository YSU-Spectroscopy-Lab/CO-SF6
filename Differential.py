import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def differential(wave, absor, df_2):
    """差分的过程"""
    # 初始化Standard_all为空数组
    diff_all = []
    OP_all = []
    concentration_all = []
    Slow_change_all = []
    # 循环处理每一列数据
    for i in range(0, len(absor)):
        absor_i = absor[i, :]
        concentration = df_2[i, :]
        # 用多项式进行拟合
        coefficients = np.polyfit(wave, absor_i, 5)
        # 构建拟合曲线方程
        poly_equation = np.poly1d(coefficients)
        # 计算整体范围内的正常值
        Slow_change = poly_equation(wave)
        # 计算校正后的光谱  差分后的光谱
        diff = absor_i - Slow_change
        OP = np.sum(abs(diff))
        diff_all.append(diff)
        OP_all.append(OP)
        concentration_all.append(concentration)
        Slow_change_all.append(Slow_change)
        # 绘制结果
        plt.figure(1)
        plt.subplot(1, 2, 1)
        plt.plot(wave, absor_i, label='Original spectrum')
        plt.plot(wave, Slow_change, label='Baseline')
        plt.subplot(1, 2, 2)
        plt.plot(wave, diff, label='Corrected spectrum')
        # plt.legend()
        plt.draw()  # 刷新图
    plt.title('The process of differentiation')
    plt.show()
    # plt.show()
    # 浓度与OP之间的关系

    plt.figure(2)
    plt.plot(concentration_all, OP_all, 'o-', label='Differential OP')
    plt.title('The relationship between concentration and differential spectral OP')
    plt.show()
    plt.close()
    # 输出浓度、OP 值

    df_absor = pd.DataFrame(absor)
    df_con = pd.DataFrame(df_2)
    df_diff = pd.DataFrame(diff_all)
    df_op = pd.DataFrame(OP_all)
    df_wave = pd.DataFrame(wave)
    df_baseline = pd.DataFrame(Slow_change_all)
    # 创建一个Excel写入器
    if len(df_diff) > 1:
        with pd.ExcelWriter('data/differential/differential.xlsx') as writer:
            df_diff.to_excel(writer, sheet_name='differential', index=False, header=False)  # 差分后的光谱
            df_baseline.to_excel(writer, sheet_name="Slow_change", index=False, header=False)
            df_absor.to_excel(writer, sheet_name='original_data', index=False, header=False)  #原始的光谱
            df_con.to_excel(writer, sheet_name='concentration', index=False, header=False)  # 浓度值
            df_op.to_excel(writer, sheet_name='OP', index=False, header=False)
            df_wave.to_excel(writer, sheet_name='wave', index=False, header=False)
    else:
        with pd.ExcelWriter('data/differential/standard_differential.xlsx') as writer:
            df_diff.to_excel(writer, sheet_name='standard_differential', index=False,
                             header=False)  # 标准光谱的差分光谱
            df_baseline.to_excel(writer, sheet_name="standard_Slow_change", index=False, header=False)
            df_wave.to_excel(writer, sheet_name='wave', index=False, header=False)

    return diff_all, Slow_change_all
