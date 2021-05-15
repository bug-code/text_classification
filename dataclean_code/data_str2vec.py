'''
created by Yang in 2020.11.1
'''
import numpy as np
from keras_preprocessing.text import Tokenizer
import re
from keras_preprocessing.sequence import pad_sequences
#读取数据集函数
def get_dataset(dataset_path):
    npzfile = np.load(dataset_path)
    return npzfile['x'] , npzfile['y']

#计算最大文本长度
def get_maxlen(train_texts , test_texts):
    maxlen = 0
    for text in train_texts:
        text = re.findall(r'\b\w+\b' , text)
        if len(text) > maxlen:
            maxlen = len(text)
    for text in test_texts:
        text = re.findall(r'\b\w+\b' , text)
        if len(text) > maxlen:
            maxlen = len(text)
    return maxlen


#文本转换为向量并填充
def texts2vec_padding(train_texts , test_texts ,maxlen):
    tokenizer = Tokenizer(num_words=10604)
    tokenizer.fit_on_texts(train_texts)
    texts_train = tokenizer.texts_to_sequences(train_texts)
    texts_test = tokenizer.texts_to_sequences(test_texts)
    #填充
    texts_train = pad_sequences(texts_train , padding='post' , maxlen=maxlen)
    texts_test = pad_sequences(texts_test , padding='post' , maxlen=maxlen)

    vocab_size = len(tokenizer.word_index)+1


    return texts_train , texts_test , vocab_size




train_dataset_path = 'F:/code/tfcode/school_code/dataclean_code/train_dataset.npz'
test_dataset_path = 'F:/code/tfcode/school_code/dataclean_code/test_dataset.npz'
#训练集文本与标签
train_x , train_y = get_dataset(train_dataset_path)
test_x , test_y = get_dataset(test_dataset_path)
#获取最大文本长度
maxlen = get_maxlen(train_x , test_x)



train_x , test_x  , vocab_size = texts2vec_padding(train_x , test_x , maxlen)

print(train_x[0])

np.savez('train_data_vec' , x=train_x , y=train_y)
np.savez('test_data_vec' , x=test_x , y=test_y)




