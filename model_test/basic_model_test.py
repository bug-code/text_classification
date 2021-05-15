'''
created by Yang in 2020.11.1
'''
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers , Sequential , datasets , optimizers,losses 
from tensorflow.keras.layers import LSTM 
import matplotlib.pyplot as plt
#读取数据集函数
def get_dataset(dataset_path):
    npzfile = np.load(dataset_path)
    return npzfile['x'] , npzfile['y']

#数据集打乱切分，组合成database
def get_database(x_train , y_train , x_test , y_test):
    train_db = tf.data.Dataset.from_tensor_slices((x_train , y_train))
    train_db=train_db.shuffle(1000).batch(128)
    #构建测试机对象，预处理，批量化
    test_db = tf.data.Dataset.from_tensor_slices((x_test , y_test))
    test_db=test_db.shuffle(1000).batch(128)
    return train_db ,test_db

#训练和测试
def train_test(model , train_dataset  ,test_dataset, epochs):
    #交叉熵损失函数类实例，带softmax函数
    CC_loss = losses.CategoricalCrossentropy(from_logits=True)
    #梯度下降优化
    optimizer = optimizers.Adam(learning_rate=0.001)
    #记录每个epoch的损失函数和准确率
    epochs_all_loss = []
    epochs_all_acc = []
    for epoch in range(epochs):
        losses_ = []
        for step , (x , y) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                # x = tf.expand_dims(x , axis =3)
                out = model(x,training=True)
                y = tf.one_hot(y,depth=5 , on_value=None , off_value=None)
                loss = CC_loss(y , out)
                aver_loss = tf.reduce_mean(loss)
                losses_.append(float(aver_loss))
            grads = tape.gradient(loss , model.trainable_variables)
            optimizer.apply_gradients(zip(grads , model.trainable_variables))
            if step%200==0:
                print("training:epoch",epoch,"step",step , 'loss',float(aver_loss) )
        aver_epoch_loss = tf.reduce_mean(losses_)
        epochs_all_loss.append(aver_epoch_loss)
        
        
        
        correct = 0
        total_samples =0
        for step , (x,y) in enumerate(test_dataset):
            # x = tf.expand_dims(x , axis =3)
            out = model(x,training=False)
            pred = tf.argmax(out , axis=-1)
            y = tf.cast(y , tf.int64)
            correct +=float( tf.reduce_sum( tf.cast( tf.equal(pred , y) , tf.float32 ) ) )
            total_samples += x.shape[0] 
        acc = correct / total_samples
        print('epoch:' , epoch , 'loss:' , float(aver_epoch_loss) , 'acc:' , float(acc),'\n')
        epochs_all_acc.append(acc)

    return epochs_all_acc , epochs_all_loss


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
    #组合成db
    traindb , test_db = get_database(train_x , train_y , test_x , test_y)


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
    layers.BatchNormalization(),
    layers.Dense(50,activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(25,activation='relu')  ,
    layers.BatchNormalization() ,
    layers.Dense(5,activation='sigmoid') ,
    ])

    #lstm模型
    lstm_model = Sequential([
        layers.Embedding(input_dim=vocab_size,
                                    output_dim=embedding_dim,
                                    input_length=max_textlen),
        LSTM(128 , return_sequences=False),
        layers.BatchNormalization() ,
        # layers.Dense(256 ,activation='relu'),
        # layers.Dense(128 ,activation='relu'),
        layers.Dense(16 ,activation='relu'),
        # layers.BatchNormalization(),
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



    
    #获取准确率和loss
    acc , loss = train_test(CNN_model, traindb , test_db , 100)
    
    #画图
    draw(loss , acc)
