import numpy as np
import pandas as pd
from Differential import differential
from Expansion_two_section_fitting import expansion_two_section_fitting
from None_filter import none_filter
from SG_filter import sg_filter
from WT_filter import wt_filter
from VMD_filter import vmd_filter
from SIN_filter import sin_filter

# 加载数据
path = 'data/Original_data_CO.xlsx'  # 替换为你的Excel文件路径

standard_absorbance = pd.read_excel(path, sheet_name='Standard_absorbance', header=None)
standard_concentration = pd.read_excel(path, sheet_name='Standard_concentration', header=None)
absorbance = pd.read_excel(path, sheet_name='absorbance', header=None)
concentration = pd.read_excel(path, sheet_name='concentration', header=None)

wave_all = np.array(absorbance)[:, 0]  # 读取波长
absor_all = np.array(absorbance)[:, 1:]  # 读取第二列到最后一  高浓度到低浓度
standard_absorbance = np.array(standard_absorbance)[:, 1]  # 读取标准光谱的列
concentration = np.array(concentration)  # 横向放置
standard_concentration = np.array(standard_concentration)
# 全都是按照列进行存储的


# 确定波长变量取值范围  并进行转置处理，按照习惯横向存放
wa_min = 2048
wa_max = 2144
id_min = np.where(wave_all >= wa_min)[0][0]
id_max = np.where(wave_all <= wa_max)[0][-1] + 1
wave = wave_all[id_min:id_max].T
absor = absor_all[id_min:id_max].T
print('absor', absor.shape)
standard_absor = standard_absorbance[id_min:id_max].reshape(398, 1).T
print('stand_absorbance', standard_absor.shape)
concentration = concentration.T  # 横向放置
print('concen', len(concentration))

# 差分处理
diff_all, Slow_change_all = differential(wave, absor, concentration)
standard_diff, standard_Slow_change = differential(wave, standard_absor, standard_concentration)
print('Differential processing completed!')

# """
Extended_all = ['Extended_20000']  # 存入的文件名
Extended = Extended_all[0]
File_all = ['two_section_fitting']
file_save = File_all[0]
filter_file = ['None_correct', 'SG_correct', 'VMD_correct', 'WT_correct', 'sin_correct']
filter_correct_save = filter_file[4]

# 数据集扩充

# concentration_all, absor_new = expansion_two_section_fitting(concentration, diff_all, wave, file_save, Extended)

print('Expansion completed!')

# 滤波处理（None/SG/VMD/WT/sin）(扩充光谱)with(测试光谱)
# Extended_spectrum = none_filter(concentration_all, absor_new, wave,file_save,filter_correct_save)
# Extended_spectrum = sg_filter(concentration_all, absor_new, wave,file_save,filter_correct_save)
# Extended_spectrum = vmd_filter(concentration_all, absor_new, wave,file_save,filter_correct_save)
# Extended_spectrum = wt_filter(concentration_all, absor_new, wave,file_save,filter_correct_save)

standard_spetrum = np.array(standard_diff)[0, :]
# Extended_spectrum = sin_filter(concentration_all, absor_new, standard_spetrum, wave, file_save, filter_correct_save)
# print('Extended data filtering completed!')

# Test_spectrum = none_filter(np.array(concentration), np.array(diff_all), wave, file_save, filter_correct_save)
# Test_spectrum = sg_filter(np.array(concentration), np.array(diff_all),  wave,file_save,filter_correct_save)
# Test_spectrum = vmd_filter(np.array(concentration), np.array(diff_all),  wave,file_save,filter_correct_save)
# Test_spectrum = wt_filter(np.array(concentration), np.array(diff_all),  wave,file_save,filter_correct_save)
Test_spectrum = sin_filter(np.array(concentration), np.array(diff_all), standard_spetrum, wave, file_save,
                           filter_correct_save)
print('Test data filtering completed!')
