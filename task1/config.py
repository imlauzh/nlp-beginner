import os

# 获取运行该文件时的执行路径
# os.path.dirname(__file__)     返回文件路径
pwd_path = os.path.abspath(os.path.dirname(__file__))
# print(pwd_path)


class Config(object):
    # 训练模型用到的路径
    dataset_path = os.path.join(pwd_path+'/data')

    train_path = os.path.join(dataset_path, 'cnews.train.txt')
    seg_train_path = os.path.join(dataset_path, 'seg.train.txt')
    # small_train_path = os.path.join(dataset_path, 'cnews.small.txt')
    stopwords_path = os.path.join(dataset_path, 'stopwords.txt')
    categories_save_path = os.path.join(dataset_path, 'categories.txt')

    # save path
    save_path = os.path.join(pwd_path+'/save')

    # tfidf model保存路径
    tfidf_model_save_path = os.path.join(save_path, 'tfidf_model.m')
    # lr save path
    lr_save_path = os.path.join(save_path+'/checkpoints', 'best_validation')

    # x
    num_epochs = 100  # 迭代轮次
    # num_classes = 10  # 类别数自动获取len(y_train[0])
    print_per_epochs = 100  # 每多少次输出
