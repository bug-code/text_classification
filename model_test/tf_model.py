
'''
created by Yang in 2020.11.1
'''
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


    train_y = tf.one_hot(train_y , on_value=None,off_value=None , depth = 5)
    test_y = tf.one_hot(test_y , on_value=None,off_value=None,depth=5)

    #数据集词袋大小
    vocab_size  =10604
    #最长文本
    max_textlen = 384
    #词向量大小
    embedding_dim  =50
    #创建文本分类基础模型
    basic_model = Sequential([
    layers.Embedding(input_dim=vocab_size , 
                    output_dim=embedding_dim,
                    input_length=max_textlen),
    layers.Flatten(),
    layers.Dense(50,activation='relu'),
    layers.Dense(25,activation='relu')  ,
    layers.Dense(5,activation='sigmoid') ,
    ])

    #lstm模型
    lstm_model = Sequential([
        layers.Embedding(input_dim=vocab_size,
                                    output_dim=embedding_dim,
                                    input_length=max_textlen),
        LSTM(128 , return_sequences=False),
        layers.BatchNormalization() ,
        layers.Dense(16 ,activation='relu'),
        layers.Dense(5,activation='relu')
    ])

    #CNN模型
    CNN_model = Sequential([
        layers.Embedding(input_dim=vocab_size,
                                    output_dim=embedding_dim,
                                    input_length=max_textlen),
        layers.Conv1D(512 ,5, activation='relu'),
        layers.GlobalMaxPooling1D(),
        layers.Dense(64 , activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(32 , activation='relu'),
        layers.Dense(5 , activation='relu')
    ])
    
    CNN_model.compile(optimizer='adam',loss='binary_crossentropy' , metrics=['accuracy'])
    CNN_model.summary()
    history = CNN_model.fit(train_x , train_y , epochs=100 , verbose=False , validation_data=(test_x , test_y),batch_size=128)
    loss  , acc =CNN_model.evaluate(train_x , train_y  , verbose=False)
    loss  , acc =CNN_model.evaluate(test_x , test_y  , verbose=False)
    draw(history.history['loss'] , history.history['accuracy'])