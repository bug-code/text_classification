'''
created by Yang in 2020.11.1
'''
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers , Sequential , metrics , losses , optimizers 
import  nltk
from nltk import word_tokenize , pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import FreqDist
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from imageio import imread
import sys
import os
import csv
import re
from collections import defaultdict
from gensim.corpora import Dictionary
import string
import numpy as np
'''
数据集预处理:
            分词--词性标注--词形归一化--去除停用词--去除特殊字符--单词大小写转换-（文本分析）
'''
#文本预处理函数

def text_pretreat(text):
    """
    对读取的文本数据进行清洗，并返回
    """
    #分词
    token_words = word_tokenize(text)
    #词性标注
    token_words = pos_tag(token_words)
    #词形归一化(指明单词词性)
    words_lematizer = []
    wordnet_lematizer = WordNetLemmatizer()
    for word, tag in token_words:
        if tag.startswith('NN'):
            word_lematizer =  wordnet_lematizer.lemmatize(word, pos='n')  # n代表名词
        elif tag.startswith('VB'): 
            word_lematizer =  wordnet_lematizer.lemmatize(word, pos='v')   # v代表动词
        elif tag.startswith('JJ'): 
            word_lematizer =  wordnet_lematizer.lemmatize(word, pos='a')   # a代表形容词
        elif tag.startswith('R'): 
            word_lematizer =  wordnet_lematizer.lemmatize(word, pos='r')   # r代表代词
        else: 
            word_lematizer =  wordnet_lematizer.lemmatize(word)
        words_lematizer.append(word_lematizer)
    #去除停用词
    cleaned_words = [word for word in words_lematizer if word not in stopwords.words('english')]
    words_list = [word for word in cleaned_words if word not in string.punctuation]
    #大小写转化
    words_lists = [x.lower() for x in words_list ]

    return words_lists

def text_show(words_lists):
    """
    文本分析
    """
    freq = FreqDist(words_lists)   
    for key,val in freq.items():
        print (str(key) + ':' + str(val))
    #可视化折线图
    freq.plot(20,cumulative=False)
    #可视化词云
    words = ' '.join(words_lists)
    wc = WordCloud().generate(words)
    plt.imshow(wc,interpolation='bilinear')
    plt.axis("off")
    plt.show()


'''
文件处理：
    读取文件，语料清洗，另存语料文件
'''
#文件读取函数
def readfile(file_path):
    fp = open(file_path , mode='r' , encoding='UTF-8' )
    content = fp.read()
    fp.close()
    return content



#语料文件夹清洗
def clean_save(read_path , save_name):
    dic = {'C':0 , 'E':1 , 'R':2 , 'T':3 , 'V':4} 
    #初步清洗训练集文本内容放在一个文件，做词频处理
    texts_infile=[]
    #暂时存放所有数据
    tmp_all_contents = []
    #存放所有数据
    texts=[]
    laybels = []
    #获取读取文件路径下的所有文件目录
    seg_dirs = os.listdir(read_path)
    for dir in seg_dirs:
        #获取子目录完整路径
        seg_dirpath = read_path + "/"+dir + "/"
        #获取子目录下所有文件
        files_list = os.listdir(seg_dirpath)    
        #遍历子目录所有文件
        for file in files_list:
            #获取文件完整目录
            file_path = seg_dirpath +"/"+ file
            #单个文本数据
            text_dataset = []
            #获取文件内容
            content = readfile(file_path).strip()
            #获取英文文本数据
            content = re.sub(r'[^a-zA-Z]'," ",content)
            #删除换行与多余空格
            content = content.replace("\r\n" , " ").strip()
            #文本处理
            content = text_pretreat(content)
            texts_infile +=content
            #结果展示
            # text_show(content)
            #追加子目录文件内容并添加标签
            tmp_data= list(content)
            tmp_laybel = dic[dir]
            text_dataset.append(tmp_data)
            text_dataset.append(tmp_laybel)            
            tmp_all_contents.append(text_dataset)
    #统计所有训练集数据词频
    frequency = defaultdict(int)
    for word in texts_infile:
        frequency[word] += 1

    #删除只出现一次的单词
    for text in tmp_all_contents:
        data = [w for w in text[0] if frequency[w] >1 and len(w) > 1 ]
        string_data = " ".join(str(i) for i in data)
        label = text[1]
        texts.append(string_data)
        laybels.append(label)
        

    np.savez(save_name , x = texts , y = laybels)


#主函数
if __name__ == "__main__":
    #训练集读取路径
    traindataset_path = "F:/code/tfcode/school_code/train/"
    #测试集读取路径
    testdataset_path = "F:/code/tfcode/school_code/test/" 
    #训练集清洗
    clean_save(traindataset_path , save_name='train_dataset' )
    clean_save(testdataset_path  , save_name='test_dataset')
