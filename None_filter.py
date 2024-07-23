import matplotlib.pyplot as plt  # 用于画图
import pandas as pd
import pickle
"None滤波"
def none_filter(concentration, absor_new, wave,file_save,filter_correct_save):
    # print('con', concentration.shape)
    # print('absor', absor_new.shape)

    OP_all = []
    concentration_all = []
    spectrum_all = []

    for j in range(0, len(absor_new)):
        y_sorted_i = absor_new[j, :]
        concentration_0 = concentration[j, :]

        y_original_spectrum = y_sorted_i
        OP = sum(abs(y_original_spectrum))
        concentration_all.append(concentration_0)
        spectrum_all.append(y_original_spectrum)
        OP_all.append(OP)

    #     # 可视化图线
    #     plt.figure(3)
    #     plt.plot(wave, y_original_spectrum, label='None y_sorted_{}'.format(j))
    #     plt.draw()  # 刷新图形
    #     # 显示曲线
    # plt.figure(4)
    # plt.plot(concentration_all, OP_all, 'o-', label='Differential OP')
    # plt.show()

    df_OP_all = pd.DataFrame(OP_all)
    df_wave = pd.DataFrame(wave)
    df_correct = pd.DataFrame(spectrum_all)
    df_concentration = pd.DataFrame(concentration_all)
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

    return spectrum_all, concentration_all
