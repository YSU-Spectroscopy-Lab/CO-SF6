import matplotlib.pyplot as plt
from CNN_structure import build_CNN_model, coeff_determination, test_data, mean_error_test, \
    train_model, Train_validation_test_data
import pickle

from sklearn.metrics import r2_score


File_all = ['two_section_fitting']
file_save = File_all[0]
filter_file = ['None_correct', 'SG_correct', 'VMD_correct', 'WT_correct', 'sin_correct']
filter_correct_save = filter_file[0]

Result_verify = ['None_verify_high', 'SG_verify_high', 'VMD_verify_high', 'WT_verify_high', 'sin_verify_high']
Result_verify = Result_verify[0]

Result_test = ['None_test_high', 'SG_test_high', 'VMD_test_high', 'WT_test_high', 'sin_test_high']
Result_test = Result_test[0]

model_parameter = ['model_None_CNN_high', 'model_SG_CNN_high', 'model_VMD_CNN_high', 'model_WT_CNN_high',
                   'model_sin_CNN_high']
model_parameter = model_parameter[0]

with open(f'data/filter/{file_save}/{filter_correct_save}.pkl', 'rb') as file:
    data_loaded = pickle.load(file)
    Extended_spectrum = data_loaded['spectrum_all'].values[10000:20000, :]
    concentration_all = data_loaded['concentration'].values[10000:20000, :]

with open(f'data/test/{file_save}/{filter_correct_save}.pkl', 'rb') as file:
    data_loaded = pickle.load(file)
    # 1ppm以上为低浓度
    Test_spectrum = data_loaded['spectrum_all'].values[6:, :]
    concentration = data_loaded['concentration'].values[6:, :]

model = build_CNN_model('Structrue.png')
optimizer = "adam"
loss = "mean_squared_error"
num = 10000
model.compile(optimizer=optimizer, loss=loss, metrics=[coeff_determination])



X_train, X_verify, x_test, y_train, y_verify, y_test = Train_validation_test_data(Extended_spectrum,
                                                                                  concentration_all, Test_spectrum,
                                                                                  concentration, num)

train_model(model, X_train, X_verify, y_train, y_verify, num, file_save, Result_verify)


model.save(f'model_parameter/{file_save}/{model_parameter}.h5')


predicted_test = test_data(model, x_test)
R2 = r2_score(y_test, predicted_test)
print('R2', R2)
MAPE = mean_error_test(predicted_test, y_test, num, file_save, Result_test)
plt.figure(1)
plt.plot(predicted_test * num, 'o', color='red', label='predicted')
plt.plot(y_test * num, 'o', color='blue', label='true')
plt.legend()
plt.show()
