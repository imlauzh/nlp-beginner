import os

import jieba
# from sklearn.externals import joblib
import joblib
import numpy as np
from sklearn import model_selection
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from config import Config

config = Config()


class DataProcess(object):
    def __init__(self, train_path, stopwords_path, seg_train_path, model_save_path=None):
        self.train_path = train_path
        self.stopwords_path = stopwords_path
        self.model_save_path = model_save_path
        self.seg_train_path = seg_train_path

    def read_data(self):
        stopwords = list()
        with open(self.train_path, encoding='utf-8') as dataset:
            data = dataset.readlines()
        print(data[0])
        with open(self.stopwords_path, encoding='utf-8') as sw:
            tmp_stopwords = sw.readlines()
        for word in tmp_stopwords:
            stopwords.append(word[:-1])
        return data, stopwords

    # 保存
    def save_file(self, data, save_path):
        with open(save_path, 'w', encoding='utf-8') as f:
            for line in data:
                f.write(line)

    def pre_data(self, data, stopwords, test_size=0.2):
        """
        预处理数据
        test_size: test数据集占的比例
        """
        label_list = list()
        text_list = list()
        jieba.setLogLevel(20)

        # 分词，标签
        if os.path.exists(self.seg_train_path):
            print("Load segmentations......")
            with open(self.seg_train_path, encoding='utf-8') as st:
                text_list = st.readlines()
            for line in data:
                label, text = line.split('\t', 1)
                label_list.append(label)
        else:
            print("jieba segmentation......")
            for line in data:
                label, text = line.split('\t', 1)
                seg_text = [word for word in jieba.cut(
                    text) if word not in stopwords]
                text_list.append(' '.join(seg_text))
                label_list.append(label)
        print(text_list[0])
        self.save_file(text_list, self.seg_train_path)
        # label one-hot
        encoder_nums = LabelEncoder()
        label_nums = encoder_nums.fit_transform(label_list)
        # 获取类别
        categories = list(encoder_nums.classes_)
        self.save_file(categories, config.categories_save_path)
        label_nums = np.array([label_nums]).T
        # print(label_nums)
        encoder_one_hot = OneHotEncoder()
        label_one_hot = encoder_one_hot.fit_transform(label_nums)
        label_one_hot = label_one_hot.toarray()
        # model_selection.train_test_split()
        # Split arrays or matrices into random train and test subsets
        # 把数组或者矩阵转成随机的train和test子集
        return model_selection.train_test_split(text_list, label_one_hot,
                                                test_size=test_size,
                                                random_state=1024)

    def get_tfidf(self, X_train, X_test):
        vectorizer = TfidfVectorizer(min_df=100)
        vectorizer.fit_transform(X_train)
        X_train_vec = vectorizer.transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        return X_train_vec, X_test_vec, vectorizer

    def provide_data(self):
        data, stopwords = self.read_data()
        X_train, X_test, y_train, y_test = self.pre_data(
            data, stopwords, test_size=0.2)
        X_train_vec, X_test_vec, vectorizer = self.get_tfidf(X_train, X_test)
        joblib.dump(vectorizer, self.model_save_path)
        # 3、提取word2vec特征参数
        return X_train_vec, X_test_vec, y_train, y_test

    def batch_iter(self, x, y, batch_size=64):
        """迭代器，将数据分批传给模型"""
        data_len = len(x)
        # 向下取整，最后加一
        num_batch = int((data_len-1)/batch_size)+1
        indices = np.random.permutation(np.arange(data_len))
        x_shuffle = x[indices]
        y_shuffle = y[indices]
        for i in range(num_batch):
            start_id = i*batch_size
            end_id = min((i+1)*batch_size, data_len)
            yield x_shuffle[start_id: end_id], y_shuffle[start_id: end_id]
