"""
扩充方法：低浓度和高浓度分成两段，分开拟合
"""

import numpy as np
import pandas as pd
import pickle


def expansion_two_section_fitting(concentration, absor, wave, file_save, Extended):
    absor = np.array(absor)
    # print('absor', absor.shape)
    concentration = concentration.squeeze()
    # print('con', concentration.shape)

    # 确定浓度变量取值范围
    concentration_mid = 1300
    ID = np.where(concentration >= concentration_mid)[0][0]
    con_low = concentration[:ID]
    con_high = concentration[ID-1:]
    # print('con_low', con_low.shape)
    # print('con_high', con_high.shape)

    absor_con_low = absor[:ID, :]
    absor_con_high = absor[ID-1:, :]
    # print('absor_con_low', absor_con_low.shape)
    # print('absor_con_high', absor_con_high.shape)

    # con_low_extend = np.linspace(min(con_low), max(con_low), 10000)
    con_low_extend = np.linspace(0, max(con_low), 10000)
    con_high_extend = np.linspace(max(con_low), max(con_high), 10000)

    # 低浓度数据扩充
    absor_low_extend = np.empty((len(con_low_extend), absor.shape[1]))
    for j, item in zip(range(0, len(con_low_extend)), con_low_extend):
        for column_index in range(0, absor_con_low.shape[1]):  # 0表示有多少行  1表示有多少列
            column = absor_con_low[:, column_index]
            coefficients = np.polyfit(con_low, column, 2)
            poly_equation = np.poly1d(coefficients)
            y = poly_equation(item)
            absor_low_extend[j, column_index] = y

    print('absor_low_extend:', absor_low_extend.shape)

    # 高浓度数据扩充
    absor_high_extend = np.empty((len(con_high_extend), absor.shape[1]))
    for i, item in zip(range(0, len(con_high_extend)), con_high_extend):
        for column_index in range(0, absor_con_high.shape[1]):  # 0表示有多少行  1表示有多少列
            column = absor_con_high[:, column_index]
            coefficients = np.polyfit(con_high, column, 4)
            poly_equation = np.poly1d(coefficients)
            y = poly_equation(item)
            absor_high_extend[i, column_index] = y

    print('absor_high_extend:', absor_high_extend.shape)

    df_concentration_low_extend = pd.DataFrame(con_low_extend)
    df_absor_low_extend = pd.DataFrame(absor_low_extend)
    df_concentration_high_extend = pd.DataFrame(con_high_extend)
    df_absor_high_extend = pd.DataFrame(absor_high_extend)
    df_wave = pd.DataFrame(wave)

    df_concentration_extend_all = pd.concat([df_concentration_low_extend, df_concentration_high_extend],
                                            ignore_index=True)
    df_absor_extend_all = pd.concat([df_absor_low_extend, df_absor_high_extend], ignore_index=True)

    # 将这三个DataFrame对象放入一个字典中
    data = {'concentration_new': df_concentration_extend_all, 'absorbance': df_absor_extend_all, 'wave': df_wave}
    with open(f'data/expansion/{file_save}/{Extended}.pkl', 'wb') as file:
        pickle.dump(data, file)

    with pd.ExcelWriter(f'data/expansion/{file_save}/{Extended}.xlsx') as writer:
        df_concentration_extend_all.to_excel(writer, sheet_name='concentration_new', index=False, header=False)  # 生成的浓度
        df_absor_extend_all.to_excel(writer, sheet_name='absorbance', index=False, header=False)  #
        df_wave.to_excel(writer, sheet_name='wave', index=False, header=False)  # 波长
    print(f"{Extended}数据扩充已完成")

    concentration_all = np.array(df_concentration_extend_all)
    absor_new = np.array(df_absor_extend_all)

    return concentration_all, absor_new
