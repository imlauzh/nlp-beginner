import time
from datetime import timedelta
from config import Config
from lr_model import LrModel
from datahelper.data_process import DataProcess
import tensorflow as tf


def get_time_dif(start_time):
    """获取已经使用的时间"""
    end_time = time.time()
    time_dif = end_time-start_time
    return timedelta(seconds=int(round(time_dif)))


def get_data():
    # 读取数据集
    print("Loading training and validation data...")
    X_train, X_test, y_train, y_test = data_get.provide_data()
    X_train = X_train.toarray()
    X_test = X_test.toarray()
    return X_train, X_test, y_train, y_test, len(X_train[0]), len(y_train[0])


def evaluate(sess, x_, y_):
    """测试集上准确率评估"""
    data_len = len(x_)
    batch_eval = data_get.batch_iter(x_, y_, batch_size=128)
    total_loss = 0
    total_acc = 0
    for batch_xs, batch_ys in batch_eval:
        batch_len = len(batch_xs)
        loss, acc = sess.run([model.loss, model.accuracy], feed_dict={
                             model.x: batch_xs, model.y_: batch_ys})
        total_loss += loss*batch_len
        total_acc += acc*batch_len
    return total_loss/data_len, total_acc/data_len


def train(X_train, X_test, y_train, y_test):
    saver = tf.train.Saver()
    # 训练模型
    print("Training and evaluating...")
    start_time = time.time()
    total_batch = 0
    best_acc_val = 0.0
    last_improved = 0
    require_improvement = 1000
    max_batch = 4000
    flag = False
    # gpu
    cfg = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    cfg.gpu_options.allow_growth = True
    with tf.Session(config=cfg) as sess:
        sess.run(tf.global_variables_initializer())
        for step in range(config.num_epochs):
            batch_train = data_get.batch_iter(X_train, y_train, batch_size=1024)
            # 迭代器获得的随机batch数据
            for batch_xs, batch_ys in batch_train:
                if total_batch % config.print_per_epochs == 0:
                    loss_train, acc_train = sess.run([model.loss, model.accuracy], feed_dict={
                                                     model.x: X_train, model.y_: y_train})
                    loss_val, acc_val = evaluate(sess, X_test, y_test)
                    if acc_val > best_acc_val:
                        best_acc_val = acc_val
                        last_improved = total_batch
                        saver.save(sess, save_path=config.lr_save_path)
                        improve_str = "*"
                    else:
                        improve_str = ""
                    time_dif = get_time_dif(start_time)
                    msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%}, ' + \
                        'Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, Time: {5} {6}'
                    print(msg.format(total_batch, loss_train, acc_train,
                                     loss_val, acc_val, time_dif, improve_str))
                sess.run(model.train_step, feed_dict={
                         model.x: batch_xs, model.y_: batch_ys})
                total_batch += 1

                if total_batch > max_batch:
                    #  轮次，提前结束训练
                    print("Too much batchs, auto-stopping...")
                    flag = True
                    break
                if total_batch-last_improved > require_improvement:
                    #  验证集准确率长期不提升，提前结束训练
                    print("No optimization for a long time, auto-stopping...")
                    flag = True
                    break
            if flag:
                break


if __name__ == "__main__":
    config = Config()  # 配置参数
    # 数据部分
    data_get = DataProcess(config.train_path, config.stopwords_path,
                           config.seg_train_path, config.tfidf_model_save_path)
    X_train, X_test, y_train, y_test, seq_length, num_classes = get_data()

    model = LrModel(config, seq_length, num_classes)
    train(X_train, X_test, y_train, y_test)
