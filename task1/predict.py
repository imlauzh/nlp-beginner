import tensorflow as tf
import os
import joblib
import jieba
from config import Config
from lr_model import LrModel


def save_file(data, save_path):
    with open(save_path, 'w', encoding='utf-8') as f:
        for line in data:
            f.write(line)
        print("save successfully")


def pre_data(dataset_path, config):
    """分词停用词"""
    stopwords = list()
    text_list = list()
    data = list()
    jieba.setLogLevel(20)
    with open(dataset_path+'/cnews.test.txt', 'r', encoding='utf-8') as f:
        test_data = f.readlines()
        for line in test_data:
            label, content = line.split('\t', 1)
            data.append(content)
    print(data[0])
    if os.path.exists(dataset_path+'/seg.test.txt'):
        print("Load segmentations......")
        with open(dataset_path+'/seg.test.txt', encoding='utf-8') as ss:
            text_list = ss.readlines()
    else:
        print("jieba segmentation......")
        with open(config.stopwords_path, 'r', encoding='utf-8') as f:
            for word in f.readlines():
                stopwords.append(word[:-1])
        for line in data:
            seg_text = jieba.cut(line)
            text = [word for word in seg_text if word not in stopwords]
            text_list.append(' '.join(text))
        save_file(text_list, dataset_path+'/seg.test.txt')
    print(text_list[0])
    return text_list


def read_categories():
    """读取类别"""
    with open(config.categories_save_path, 'r', encoding='utf-8') as f:
        categories = f.readlines()
    return categories[0].split('|')


def predict_line(data, categories):
    """预测结果"""
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path=config.lr_save_path)
    y_pred_cls = session.run(model.y_pred_cls, feed_dict={model.x: data})
    print(y_pred_cls[:100])
    return [categories[i] for i in y_pred_cls]


if __name__ == "__main__":
    config = Config()
    line = pre_data(config.dataset_path, config)
    tfidf_model = joblib.load(config.tfidf_model_save_path)
    X_test = tfidf_model.transform(line).toarray()
    categories = read_categories()
    model = LrModel(config, len(X_test[0]), len(categories))
    print(predict_line(X_test, categories))
