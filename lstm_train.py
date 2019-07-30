import os
import time
import shutil
import numpy as np
import tensorflow as tf
import lstm_model
import urllib.request
from bs4 import BeautifulSoup

# 在tensorflow的log日志等级如下：
# - 0：显示所有日志（默认等级）
# - 1：显示info、warning和error日志
# - 2：显示warning和error信息
# - 3：显示error日志信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
BASEDIR = os.path.dirname(os.path.abspath(__file__))

train_batch_size = 256
test_batch_size = 30
keep_prob = 0.7
regularaztion_rate = 1e-5
learning_rate = 0.01
n_h1 = 100
n_hidden = 80
n_input = 7
n_steps = 9
n_classnum = 34
n_output = 7


# 爬取彩票数据
def scrapedata(filename):
    useragent = 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
    headers = {'User-Agent': useragent}
    url = 'http://kaijiang.zhcw.com/zhcw/html/ssq/list.html'
    req = urllib.request.Request(url, headers=headers)
    html = urllib.request.urlopen(req, timeout=10)
    bsObj = BeautifulSoup(html, 'html.parser', from_encoding='gb18030')
    html.close()
    pageid = int(bsObj.find('p', {'class': 'pg'}).find('strong').text)
    f = open(os.path.join(BASEDIR, filename), 'w')
    for i in range(1, pageid + 1):
        url = 'http://kaijiang.zhcw.com/zhcw/inc/ssq/ssq_wqhg.jsp?pageNum=%d' % i
        print("scrape url:%s" % url)
        req = urllib.request.Request(url, headers=headers)
        html = urllib.request.urlopen(req, timeout=30)
        bsObj = BeautifulSoup(html, 'html.parser', from_encoding='gb18030')
        html.close()
        num = bsObj.findAll('tr')
        for i in num[2:-1]:
            j = i.findAll('td')
            f.write(j[0].text.replace('-', ''))
            for x in j[2].findAll("em"):
                a = int(x.text)
                f.write(' ' + str(a))
            f.write('\n')
    f.close()


# 将彩票数据按照日期降序，并去掉日期列数据
def read_data(filename):
    np_data = np.loadtxt(os.path.join(BASEDIR, filename), dtype=np.float32)
    np_data = np_data[:, 1:]
    data_list = np_data[::-1]
    np_data = data_list[:-n_steps]
    return np_data


# 将彩票数据重新排列成n_steps深度，并将后n_steps以后的数据作为标签数据
def preprocess_data(data, n_steps):
    tmp_data = []
    for i in range(n_steps):
        tmp_data.append(data[i:-n_steps + i])
    datas = np.concatenate(tmp_data, axis=1)
    labels = data[n_steps:]
    return datas, labels


# 将数据转化为batch
def get_batch(data, label, batch_size):
    input_queue = tf.train.slice_input_producer([data, label])
    label = input_queue[1]
    data_contents = input_queue[0]
    date_batch, label_batch = tf.train.batch([data_contents, label], batch_size=batch_size, num_threads=4, capacity=batch_size * 4 + 2000)
    return date_batch, label_batch


# 开始训练模型
def run_training(filename, log):
    scrapedata(filename)
    log_dir = os.path.join(BASEDIR, log)
    # 删除目录下的文件
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    files = os.listdir(log_dir)
    for i in files:
        if os.path.isfile(os.path.join(log_dir, i)):
            os.remove(os.path.join(log_dir, i))
        else:
            # shutil库为python内置库，是一个对文件及文件夹高级操作的库,
            # 递归删除文件夹
            shutil.rmtree(os.path.join(log_dir, i))
    np_data = read_data(filename)
    data, data_label = preprocess_data(np_data, n_steps)
    le = len(data) * 9 // 10
    with tf.Graph().as_default():
        train = data[:le] / 33.
        test = data[le:] / 33.
        train_label = tf.one_hot(data_label[:le], n_classnum)
        test_label = tf.one_hot(data_label[le:], n_classnum)
        train_batch, train_label_batch = get_batch(train, train_label, train_batch_size)
        test_batch, test_label_batch = get_batch(test, test_label, test_batch_size)
        # 使用is_traning来判断是训练还是验证，并传递不通的keep_prob值
        is_traning = tf.placeholder(tf.bool, shape=())
        # tf.cond()类似于c语言中的if...else...，用来控制数据流向
        X_ = tf.cond(is_traning, lambda: train_batch, lambda: test_batch)
        Y_ = tf.cond(is_traning, lambda: train_label_batch, lambda: test_label_batch)
        keep_prob_place = tf.cond(is_traning, lambda: keep_prob, lambda: 1.0)
        batch_size_place = tf.cond(is_traning, lambda: train_batch_size, lambda: test_batch_size)
        regularizer = tf.contrib.layers.l2_regularizer(scale=regularaztion_rate)
        # 使用cnn网络
        fc1 = lstm_model.fc_net('fc1', X_, [n_input * n_steps, 100], regularizer=False, is_traning=is_traning)
        fc1 = tf.nn.relu(fc1)
        reshape_fc1 = tf.reshape(fc1, [-1, 10, 10, 1])
        conv1 = lstm_model.conv_net('conv1', reshape_fc1, [3, 3, 1, 16], [1, 4, 4, 1])
        pool1 = lstm_model.max_pool('pool1', conv1, [1, 3, 3, 1], [1, 2, 2, 1],)
        rule1 = tf.nn.relu(pool1)
        conv2 = lstm_model.conv_net('conv2', rule1, [3, 3, 16, 32], [1, 4, 4, 1])
        pool2 = lstm_model.max_pool('pool2', conv2, [1, 3, 3, 1], [1, 2, 2, 1],)
        rule2 = tf.nn.relu(pool2)
        conv3 = lstm_model.conv_net('conv3', rule2, [3, 3, 32, 64], [1, 4, 4, 1])
        pool3 = lstm_model.max_pool('pool3', conv3, [1, 3, 3, 1], [1, 2, 2, 1],)
        rule3 = tf.nn.relu(pool3)
        conv4 = lstm_model.conv_net('conv4', rule3, [3, 3, 64, 256], [1, 4, 4, 1])
        pool4 = lstm_model.max_pool('pool4', conv4, [1, 3, 3, 1], [1, 2, 2, 1],)
        rule4 = tf.nn.relu(pool4)
        reshape_fc2 = tf.reshape(rule4, [-1, 256])
        fc2 = lstm_model.fc_net('fc2', reshape_fc2, [256, n_output * n_classnum])
        # 使用lstm网络
        # lstm_1 = lstm_model.lstm_net('lstm_1', X_, n_hidden=n_hidden, n_steps=n_steps, n_input=n_input, n_output=n_classnum * n_output, batch_size=batch_size_place,
        #                              keep_prob=keep_prob_place, regularizer=False, is_traning=is_traning)
        logits = tf.reshape(fc2, [-1, n_output, n_classnum])
        loss = lstm_model.losses(logits, Y_, n_classnum)

        op = lstm_model.trainning(loss, learning_rate)
        acc = lstm_model.evaluation(logits, Y_, n_classnum)
        # 合并摘要，包含所有输入摘要的值
        summary_op = tf.summary.merge_all()
        # 使用GPU训练
        config = tf.ConfigProto()
        # 占用GPU70%的显存,超出部分使用内存
        config.gpu_options.per_process_gpu_memory_fraction = 0.8
        sess = tf.Session(config=config)
        train_writer = tf.summary.FileWriter(os.path.join(log_dir, 'train'), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(log_dir, 'test'))
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        # 协调器，协调线程间的关系可以视一种信号量，用来做同步
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
                # tmp = sess.run(loss, feed_dict={is_traning: True})
                # print(tmp, train_label_data.shape)
                # return
                _, out_loss, out_acc = sess.run([op, loss, acc], feed_dict={is_traning: True})
                duration = time.time() - start_time
                if step % 100 == 0 and step % 1000 != 0:
                    summary_str_train = sess.run(summary_op, feed_dict={is_traning: True})
                    train_writer.add_summary(summary_str_train, global_step=step)
                    summary_str_test = sess.run(summary_op, feed_dict={is_traning: False})
                    test_writer.add_summary(summary_str_test, global_step=step)
                    print('Step %d, train loss = %.5f, train acc = %.5f train time = %.5f' % (step, out_loss, out_acc, duration))
                    # print('Step %d, train loss = %.5f, train time = %.5f' % (step, (los_fc + los_lstm),  duration))
                if step % 1000 == 0:
                    out_loss_, out_acc_ = sess.run([loss, acc], feed_dict={is_traning: False})
                    # if out_acc - out_acc_ > 0.05:
                    #     break
                    print('Step %d, train loss = %.5f, train acc = %.5f, test loss = %.5f,test acc = %.5f train time = %.5f' % (step, out_loss, out_acc, out_loss_, out_acc_, duration))
                    checkpoint_path = os.path.join(log_dir, 'model.ckpt')
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
def check_lstm(data, log):
    log_dir = os.path.join(BASEDIR, log)
    npdata = np.reshape(data, [1, n_input * n_steps]) / 33.
    with tf.Graph().as_default():
        X_ = tf.placeholder(tf.float32, shape=[1, n_input * n_steps])
        fc1 = lstm_model.fc_net('fc1', X_, [n_input * n_steps, 100], regularizer=False)
        fc1 = tf.nn.relu(fc1)
        reshape_fc1 = tf.reshape(fc1, [-1, 10, 10, 1])
        conv1 = lstm_model.conv_net('conv1', reshape_fc1, [3, 3, 1, 16], [1, 4, 4, 1])
        pool1 = lstm_model.max_pool('pool1', conv1, [1, 3, 3, 1], [1, 2, 2, 1],)
        rule1 = tf.nn.relu(pool1)
        conv2 = lstm_model.conv_net('conv2', rule1, [3, 3, 16, 32], [1, 4, 4, 1])
        pool2 = lstm_model.max_pool('pool2', conv2, [1, 3, 3, 1], [1, 2, 2, 1],)
        rule2 = tf.nn.relu(pool2)
        conv3 = lstm_model.conv_net('conv3', rule2, [3, 3, 32, 64], [1, 4, 4, 1])
        pool3 = lstm_model.max_pool('pool3', conv3, [1, 3, 3, 1], [1, 2, 2, 1],)
        rule3 = tf.nn.relu(pool3)
        conv4 = lstm_model.conv_net('conv4', rule3, [3, 3, 64, 256], [1, 4, 4, 1])
        pool4 = lstm_model.max_pool('pool4', conv4, [1, 3, 3, 1], [1, 2, 2, 1],)
        rule4 = tf.nn.relu(pool4)
        reshape_fc2 = tf.reshape(rule4, [-1, 256])
        fc2 = lstm_model.fc_net('fc2', reshape_fc2, [256, n_output * n_classnum])
        check_logits = tf.reshape(fc2, [-1, n_output, n_classnum])
        check_logits = tf.nn.softmax(check_logits)
        # config = tf.ConfigProto()
        # 占用GPU70%的显存,超出部分使用内存
        # config.gpu_options.per_process_gpu_memory_fraction = 0.8
        saver = tf.train.Saver()
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(log_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('.')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                print('No checkpoint file found')
            prediction = sess.run(check_logits, feed_dict={X_: npdata})
            outputdata = prediction.argmax(2)
        return outputdata


if __name__ == "__main__":
    # 训练模型
    run_training('aa.txt', 'log')
    # 验证模型
    # datas = read_data('aa.txt')
    # data, data_label = preprocess_data(datas, n_steps)
    # le = len(data) * 9 // 10
    # test_data = data[le:]
    # test_label = data_label[le:]
    # for i in range(len(test_data)):
    #     logits = check_lstm(test_data[i], 'log')
    #     logits_mean = np.mean(np.equal(logits[0], test_label[i]))
    #     print("logits mean: ", logits_mean, "logits: ", logits, "real: ", test_label[i])
