import os
import time
import shutil
import numpy as np
import tensorflow as tf
import model_lstm

# 在tensorflow的log日志等级如下：
# - 0：显示所有日志（默认等级）
# - 1：显示info、warning和error日志
# - 2：显示warning和error信息
# - 3：显示error日志信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

train_batch_size = 256
test_batch_size = 30
keep_prob = 0.5
regularaztion_rate = 1e-3
learning_rate = 0.001
n_h1 = 50
n_hidden = 100
n_input = 7
n_steps = 1
n_classnum = 34
n_output = 7


def read_data(filename):
    with open(filename) as f:
        data = f.readline()
    data_list = data.split()
    np_data = np.array(data_list[: -n_input], np.float32)
    np_label = np.array(data_list[n_input:], np.int32)
    input_data = np.reshape(np_data, [-1, n_input])
    input_label = np.reshape(np_label, [-1, n_input])
    return input_data, input_label


def get_batch(data, label, batch_size):
    input_queue = tf.train.slice_input_producer([data, label])
    label = input_queue[1]
    data_contents = input_queue[0]
    date_batch, label_batch = tf.train.batch([data_contents, label], batch_size=batch_size, num_threads=4, capacity=batch_size * 4 + 2000)
    return date_batch, label_batch


def run_training(filename, logs_dir):
    # 删除目录下的文件
    if not os.path.exists(logs_dir):
        os.mkdir(logs_dir)
    files = os.listdir(logs_dir)
    for i in files:
        if os.path.isfile(os.path.join(logs_dir, i)):
            os.remove(os.path.join(logs_dir, i))
        else:
            # shutil库为python内置库，是一个对文件及文件夹高级操作的库,
            # 递归删除文件夹
            shutil.rmtree(os.path.join(logs_dir, i))
    data, data_label = read_data(filename)
    with tf.Graph().as_default():
        train = data[:2330] / 33.
        test = data[2330:] / 33.
        train_label = tf.one_hot(data_label[:2330], n_classnum)
        test_label = tf.one_hot(data_label[2330:], n_classnum)
        train_batch, train_label_batch = get_batch(train, train_label, train_batch_size)
        test_batch, test_label_batch = get_batch(test, test_label, test_batch_size)
        # X_ = tf.placeholder(tf.float32, [batch_size, n_input])
        # Y_ = tf.placeholder(tf.float32, [batch_size, n_output, n_classnum])
        # 使用is_traning来判断是训练还是验证，并传递不通的keep_prob值
        is_traning = tf.placeholder(tf.bool, shape=())
        # tf.cond()类似于c语言中的if...else...，用来控制数据流向
        X_ = tf.cond(is_traning, lambda: train_batch, lambda: test_batch)
        Y_ = tf.cond(is_traning, lambda: train_label_batch, lambda: test_label_batch)
        keep_prob_place = tf.cond(is_traning, lambda: keep_prob, lambda: 1.0)
        batch_size_place = tf.cond(is_traning, lambda: train_batch_size, lambda: test_batch_size)
        regularizer = tf.contrib.layers.l2_regularizer(scale=regularaztion_rate)
        fc1 = model_lstm.fc_net('fc1', X_, w_shape=[n_input, n_h1], regularizer=regularizer, is_traning=is_traning)
        fc1 = tf.nn.relu(fc1)
        fc1 = tf.nn.dropout(fc1, keep_prob_place)
        lstm_1 = model_lstm.lstm_net('lstm_1', fc1, n_hidden=n_h1, n_steps=1, batch_size=batch_size_place,
                                     keep_prob=keep_prob_place, regularizer=regularizer, is_traning=is_traning)
        fc2 = model_lstm.fc_net('fc2', lstm_1, [n_h1, n_hidden], regularizer=regularizer, is_traning=is_traning)
        fc2 = tf.nn.relu(fc2)
        fc2 = tf.nn.dropout(fc2, keep_prob_place)
        lstm_2 = model_lstm.lstm_net('lstm_2', fc2, n_hidden, n_steps, batch_size=batch_size_place,
                                     keep_prob=keep_prob_place, regularizer=regularizer, is_traning=is_traning)
        fc3 = model_lstm.fc_net('fc3', lstm_2, [n_hidden * n_steps, n_output * n_classnum], regularizer, is_traning)
        loss = model_lstm.losses(fc3, Y_, n_classnum)
        op = model_lstm.trainning(loss, learning_rate)
        acc = model_lstm.evaluation(fc3, Y_, n_classnum)
        # 合并摘要，包含所有输入摘要的值
        summary_op = tf.summary.merge_all()
        # 使用GPU训练
        config = tf.ConfigProto()
        # 占用GPU70%的显存,超出部分使用内存
        config.gpu_options.per_process_gpu_memory_fraction = 0.8
        sess = tf.Session(config=config)
        train_writer = tf.summary.FileWriter(os.path.join(logs_dir, 'train'), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(logs_dir, 'test'), sess.graph)
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        # 协调器，协调线程间的关系可以视为一种信号量，用来做同步
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            step = 0
            print('start training data...')
            while True:
                if coord.should_stop():
                    break
                start_time = time.time()
                step += 1
                train_data, train_label_data = sess.run([train_batch, train_label_batch])
                test_data, test_label_data = sess.run([test_batch, test_label_batch])
                # tmp = sess.run(fc3, feed_dict={X_: train_data, Y_: train_label_data})
                # print(tmp.shape)
                # return
                _, out_loss, out_acc = sess.run([op, loss, acc], feed_dict={is_traning: True})
                duration = time.time() - start_time
                if step % 100 == 0 and step % 1000 != 0:
                    summary_str_train = sess.run(summary_op, feed_dict={is_traning: True})
                    train_writer.add_summary(summary_str_train, global_step=step)
                    summary_str_test = sess.run(summary_op, feed_dict={is_traning: False})
                    test_writer.add_summary(summary_str_test, global_step=step)
                    print('Step %d, train loss = %.5f, train acc = %.5f train time = %.5f' % (step, out_loss, out_acc, duration))
                if step % 1000 == 0:
                    out_loss_, out_acc_ = sess.run([loss, acc], feed_dict={is_traning: False})
                    print('Step %d, train loss = %.5f, train acc = %.5f, test loss = %.5f,test acc = %.5f train time = %.5f' % (step, out_loss, out_acc, out_loss_, out_acc_, duration))
                    checkpoint_path = os.path.join(logs_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)
        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            # 通知其他线程关闭
            coord.request_stop()
        # join操作等待其他线程结束，其他所有线程关闭之后，这一函数才能返回
        coord.join(threads)
        sess.close()


# 验证模型
def check_lstm(data, logs_dir):
    npdata = np.reshape(data, [1, n_input]) / 33.
    with tf.Graph().as_default():
        X = tf.placeholder(tf.float32, shape=[1, n_input])
        # check_logits = lstm_net(X, batch_size=1, keep_prob=1)
        fc1 = model_lstm.fc_net('fc1', X_, w_shape=[n_input, n_h1])
        fc1 = tf.nn.relu(fc1)
        lstm_1 = model_lstm.lstm_net('lstm_1', fc1, n_hidden=n_h1, n_steps=1, batch_size=1)
        fc2 = model_lstm.fc_net('fc2', lstm_1, [n_h1, n_hidden])
        fc2 = tf.nn.relu(fc2)
        lstm_2 = model_lstm.lstm_net('lstm_2', fc2, n_hidden, n_steps, batch_size=1,)
        fc3 = model_lstm.fc_net('fc3', lstm_2, [n_hidden * n_steps, n_output * n_classnum])
        check_logits = tf.nn.softmax(fc3)
        # config = tf.ConfigProto()
        # 占用GPU70%的显存,超出部分使用内存
        # config.gpu_options.per_process_gpu_memory_fraction = 0.8
        saver = tf.train.Saver()
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(logs_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('.')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                print('No checkpoint file found')
            prediction = sess.run(check_logits, feed_dict={X: npdata})
            outputdata = prediction.reshape(-1, class_num).argmax(1).tolist()
        return outputdata


if __name__ == "__main__":
    run_training('E:\\aa_new.txt', 'E:\\log\\lstm_cp')
