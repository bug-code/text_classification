from bert_sklearn import BertClassifier
from bert_sklearn import BertRegressor
from bert_sklearn import BertTokenClassifier
from bert_sklearn import load_model
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers , Sequential , datasets , optimizers,losses 
from tensorflow.keras.layers import LSTM 
import matplotlib.pyplot as plt
import random

#读取数据集函数
def get_dataset(dataset_path):
    npzfile = np.load(dataset_path)
    return npzfile['x'] , npzfile['y']

#画图函数
def draw(epoch_sumloss , epoch_acc):
    x=[i for i in range(len(epoch_sumloss))]
    #左纵坐标
    fig , ax1 = plt.subplots()
    color = 'red'
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss' , color=color)
    ax1.plot(x , epoch_sumloss , color=color)
    ax1.tick_params(axis='y', labelcolor= color)

    ax2=ax1.twinx()
    color1='blue'
    ax2.set_ylabel('acc',color=color1)
    ax2.plot(x , epoch_acc , color=color1)
    ax2.tick_params(axis='y' , labelcolor=color1)

    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    #读取路径
    train_path = 'F:/code/tfcode/school_code/dataclean_code/train_data_vec.npz'
    test_path = 'F:/code/tfcode/school_code/dataclean_code/test_data_vec.npz'
    #读取训练集与测试集
    train_x , train_y = get_dataset(train_path)
    test_x , test_y = get_dataset(test_path)
    
    seed = 1234
    random.seed(seed)
    random.shuffle(train_x )
    random.seed(seed)
    random.shuffle(train_y)

    seed = 2143
    random.seed(seed)
    random.shuffle(test_x )
    random.seed(seed)
    random.shuffle(test_y)

    model = BertClassifier()
    model.fit(train_x , train_y)

    pre_y = model.predict(test_x)

    score = model.score(pre_y , test_y)

    print(score)
