import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
from keras.models import Model
from keras.utils import plot_model
from keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, Input
from keras import backend as K
import torch
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import Callback



def build_CNN_model(model_structure):
    input1 = Input(shape=(398, 1))
    conv_layer1_1 = Conv1D(128, 5, strides=2, activation='relu')(input1)
    max_layer1_1 = MaxPooling1D(3)(conv_layer1_1)
    conv_layer1_2 = Conv1D(64, 5, strides=2, activation='relu')(max_layer1_1)
    max_layer1_2 = MaxPooling1D(3)(conv_layer1_2)
    conv_layer1_3 = Conv1D(32, 3, strides=2, activation='relu')(max_layer1_2)
    max_layer1_3 = MaxPooling1D(3)(conv_layer1_3)
    flatten = Flatten()(max_layer1_3)
    f1 = Dense(1, activation='linear', name='prediction_one')(flatten)
    model = Model(outputs=f1, inputs=input1)
    model.summary()
    plot_model(model, to_file=model_structure, show_shapes=True)  # Printed model structure
    return model


# Predicted data
def verify_data(model, X_verify):
    # model.compile(optimizer=optimizer, loss=loss, metrics=[coeff_determination])
    predicted = model.predict(X_verify)
    return predicted


def test_data(model, x_test):
    # model.compile(optimizer=optimizer, loss=loss, metrics=[coeff_determination])
    predicted = model.predict(x_test)
    return predicted


# Custom metric function, determination factor R_Squares
def coeff_determination(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))

    return (1 - SS_res / (SS_tot + K.epsilon()))
    # return R2


def mean_error_verify(predicted, y_verify, num, fitting, Result_verify):
    predicted = np.reshape(predicted, (len(predicted), 1))
    y_test_size = y_verify
    predicted = np.array(predicted) * num
    y_verify_size = np.array(y_test_size) * num
    result_0 = abs(predicted - y_verify_size)
    result = np.mean(abs((predicted - y_verify_size) / y_verify_size)) * 100
    print("MPAE:{:.2f}%".format(result))

    df_predicted = pd.DataFrame(predicted)
    df_y_verify_size = pd.DataFrame(y_verify_size)
    df_result_0 = pd.DataFrame(result_0)
    with pd.ExcelWriter(f'data/result/{fitting}/{Result_verify}.xlsx') as writer:
        df_y_verify_size.to_excel(writer, sheet_name='concentration', startcol=0, index=False, header=False)
        df_predicted.to_excel(writer, sheet_name='concentration', startcol=1, index=False, header=False)
        df_result_0.to_excel(writer, sheet_name='concentration', startcol=2, index=False, header=False)
    return result


def mean_error_test(predicted, y_test, num, fitting, Result_test):
    predicted = np.reshape(predicted, (len(predicted), 1))
    y_test_size = y_test
    predicted = np.array(predicted) * num
    y_test_size = np.array(y_test_size) * num
    result_0 = abs(predicted - y_test_size)
    result = np.mean(abs((predicted - y_test_size) / y_test_size)) * 100
    print("MPAE:{:.2f}%".format(result))

    df_predicted = pd.DataFrame(predicted)
    df_y_test_size = pd.DataFrame(y_test_size)
    df_result_0 = pd.DataFrame(result_0)
    with pd.ExcelWriter(f'data/result/{fitting}/{Result_test}.xlsx') as writer:
        df_y_test_size.to_excel(writer, sheet_name='concentration', startcol=0, index=False, header=False)
        df_predicted.to_excel(writer, sheet_name='concentration', startcol=1, index=False, header=False)
        df_result_0.to_excel(writer, sheet_name='concentration', startcol=2, index=False, header=False)
    return result


def train_model(model, X_train, X_verify, y_train, y_verify, num, fitting, Result_verify):
    history = model.fit(X_train, y_train,
                        batch_size=256,
                        epochs=100,  #1000
                        validation_data=(X_verify, y_verify),
                        # callbacks=[early_stopping]

                        )
    predicted = verify_data(model, X_train)
    MAPE_train = mean_error_verify(predicted, y_train, num, fitting, Result_verify)

    predicted = verify_data(model, X_verify)
    MAPE_verify = mean_error_verify(predicted, y_verify, num, fitting, Result_verify)

    return MAPE_train, MAPE_verify


def Train_validation_test_data(x, y, x_test, y_test, num):
    x = np.expand_dims(x.astype(float), axis=2)
    y = y / num
    X_train, X_verify, y_train, y_verify = train_test_split(x, y, test_size=0.1, random_state=20)
    x_test = np.expand_dims(np.array(x_test).astype(float), axis=2)
    y_test = torch.tensor(y_test).float() / num

    return X_train, X_verify, x_test, y_train, y_verify, y_test
