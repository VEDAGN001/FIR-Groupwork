import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from abc import ABC
from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.utils import Sequence

input_days = 21
predict_days = 5


class Generator(Sequence, ABC):
    def __init__(self, data_, num_train, num_predict):
        self.data = data_
        self.num_train = num_train
        self.num_predict = num_predict
        self.len = len(data_) - num_train - num_predict + 1

    def __getitem__(self, bach_index):
        t = bach_index + self.num_train
        data_x = self.data[bach_index: t]
        data_y = self.data[t: t + self.num_predict]
        return [np.array([[data_x]])], [np.array([data_y])]

    def __len__(self):
        return self.len


def build_model(n_lstm_, n_layer_, n_neuron_, activation_):
    inputs = [layers.Input((1, input_days))]
    x = inputs[0]
    x = layers.LSTM(n_lstm_)(x)
    for i in range(n_layer_):
        x = layers.Dense(n_neuron_, activation_)(x)
    outputs = [layers.Dense(predict_days)(x)]
    model_ = Model(inputs, outputs)
    model_.compile(optimizers.Adam(1e-6), 'mse', ['mse'])
    print(model_.summary())
    return model_


# 读取数据
data = pd.read_csv('BABA3.17.csv')['Close'].values
# 划分训练集和测试集，90%训练，10%测试
train_data = Generator(data[:len(data) * 9 // 10], input_days, predict_days)
test_data = Generator(data[len(data) * 9 // 10:], input_days, predict_days)
# 模型参数
activation = 'relu'
epochs = 80
history = []
for n_lstm in (32,64,128):
    for n_layer in (3,6,9):
        for n_neuron in (10,20,50):
            # 构建模型
            model = build_model(n_lstm, n_layer, n_neuron, activation)
            # 训练模型
            history.append((n_lstm, n_layer, n_neuron, model.fit(train_data, epochs=epochs).history['mse'][-1]))
            # 使用最后21天的数据预测未来5天
            predict = model.predict(np.array([[data[-21:]]]))
            # 绘制图形
            x = pd.read_csv('BABA3.17.csv')['Date'].values[-21:]
            x = np.concatenate([x, np.array(['2022-03-18', '2022-03-21', '2022-03-22', '2022-03-23', '2022-03-24'])])
            y_input = data[-21:]
            y_pred = predict[0]
            plt.figure(figsize=(6.4 * 3, 4.8 * 2.1))
            plt.xticks(rotation=90)
            plt.plot(x[:21], y_input, color='black', label="Real Price")
            plt.plot(x[21:], y_pred, color='green', marker='o', markersize=5, label="Predicted Stock Price")
            plt.legend()
            plt.xlabel("Time")
            plt.ylabel("Stock Price")
            plt.show()
