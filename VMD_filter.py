import matplotlib.pyplot as plt
from vmdpy import VMD
import pandas as pd
import numpy as np
import pickle


def vmd_filter(concentration, absor_new, wave,file_save,filter_correct_save):
    print('con', concentration.shape)
    print('absor', absor_new.shape)


    # VMD参数
    alpha = 20  # moderate bandwidth constraint
    tau = 0.  # noise-tolerance (no strict fidelity enforcement)
    K = 5  # 5 modes
    DC = False  # no DC part imposed
    init = 0  # initialize omegas uniformly
    tol = 1e-7

    # 进行VMD变换
    OP_all = []
    concentration_all = []
    spectrum_all = []

    for j in range(0, len(absor_new)):
        y_sorted_i = absor_new[j, :]
        concentration_0 = concentration[j, :]
        # VMD分解
        u, u_hat, omega = VMD(y_sorted_i, alpha, tau, K, DC, init, tol)
        # 检查 u 的形状并转置（如果需要）
        if u.shape[0] == K:
            u = u.T

        # # 绘制 VMD 分解后的每个模式
        # for i in range(0,K):
        #     plt.figure(6)
        #     plt.subplot(K, 1, i+1)
        #     plt.plot(wave,u[:, i])
        #     # plt.title('Mode {}'.format(i + 1))
        #     plt.draw()
        # plt.show()

        selected_U = [0,1,2]  # 选择要叠加的u 的索引，这里选择了第1、2和4个IMF

        # 对选定的 IMFs 进行累加，重构信号
        y_spectrum = np.sum([u[:, i] for i in selected_U], axis=0)
        y_spectrum = y_spectrum[:len(y_sorted_i)]
        y_original_spectrum = y_spectrum
        OP = sum(abs(y_original_spectrum))
        concentration_all.append(concentration_0)
        spectrum_all.append(y_original_spectrum)
        OP_all.append(OP)

    #     plt.figure(3)
    #     plt.plot(wave, y_sorted_i, label='f:orignal y_sorted_{}'.format(j))
    #     plt.figure(4)
    #     plt.plot(wave, y_original_spectrum, label='f: VMD y_sorted_{}'.format(j))
    #     plt.legend()
    #     plt.draw()  # 刷新图形
    # plt.figure(5)
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
