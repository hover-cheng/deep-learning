import os
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import time

BATCH_SIZE = 16
CAPACITY = 2000 + 4 * BATCH_SIZE
REGULARAZTION_RATE = 0.001
LEARNING_RATE = 0.0001
KEEP_PROB = 0.75

# LSTM中隐藏层的数量
N_HIDDEN = 2000
# LSTM的层数
LAYER_NUM = 3
# 输入词语的个数
N_INPUT = 7
# 即每做一次预测,需要先输入1行
N_STEPS = 1
# 输出类别的数量
CLASS_NUM = 33
# 输出类别的个数
OUTPUTNUM = 7

# 在tensorflow的log日志等级如下：
# - 0：显示所有日志（默认等级）
# - 1：显示info、warning和error日志
# - 2：显示warning和error信息
# - 3：显示error日志信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class lstm(object):
    def __init__(self, LEARNING_RATE, REGULARAZTION_RATE, BATCH_SIZE, CAPACITY, N_INPUT, N_STEPS, N_HIDDEN, LAYER_NUM, CLASS_NUM, OUTPUTNUM=1, KEEP_PROB=1):
        self.learning_rate = LEARNING_RATE
        self.regularaztion_rate = REGULARAZTION_RATE
        self.batch_size = BATCH_SIZE
        self.capacity = CAPACITY
        self.n_input = N_INPUT
        self.n_steps = N_STEPS
        self.n_hidden = N_HIDDEN
        self.layer_num = LAYER_NUM
        self.class_num = CLASS_NUM
        self.outputnum = OUTPUTNUM
        self.keep_prob = KEEP_PROB

    # 读取数据，转换成向量
    def read_data(self, filename):
        with open(filename) as f:
            data = f.readline()
        data_list = data.split()
        np_data = np.array(data_list[: -self.n_input], np.float32)
        init_data = [0 for x in range(self.class_num)]
        list_label = []
        # 转换程one-hot模式，并去掉源数据中的0
        for i in data_list[self.n_input:]:
            tmp = init_data[:]
            tmp[int(i) - 1] = 1
            list_label.append(tmp)
        input_data = np.reshape(np_data, [-1, self.n_input])
        input_label = np.reshape(np.concatenate(np.array(list_label, np.float32)), [-1, self.class_num * self.outputnum])
        input_data = (input_data - input_data.min()) / (input_data.max() - input_data.min())
        return input_data, input_label

    # 生成batch
    def get_batch(self, data, label):
        input_queue = tf.train.slice_input_producer([data, label])
        label = input_queue[1]
        data_contents = input_queue[0]
        date_batch, label_batch = tf.train.shuffle_batch([data_contents, label], batch_size=self.batch_size, num_threads=4, capacity=self.capacity, min_after_dequeue=2 * self.batch_size)
        date_batch = tf.reshape(date_batch, [self.batch_size, self.n_input, self.n_steps])
        label_batch = tf.reshape(label_batch, [self.batch_size, self.class_num * self.outputnum])
        return date_batch, label_batch

    # 定义LSTM网络模型
    def inference(self, data, batch_size=None, keep_prob=None, regularizer=None):
        with tf.variable_scope('lstm', reuse=tf.AUTO_REUSE):
            # 将输入的数据转成2维进行计算，计算后的结果作为隐藏层输入
            reshaped = tf.reshape(data, [-1, self.n_input])
            weight_in = tf.Variable(tf.random_normal([self.n_input, self.n_hidden]))
            weight_out = tf.Variable(tf.random_normal([self.n_hidden, self.class_num * self.outputnum]))
            biase_in = tf.Variable(tf.constant(0.1, shape=[self.n_hidden, ]))
            biase_out = tf.Variable(tf.constant(0.1, shape=[self.class_num * self.outputnum, ]))
            X_in = tf.matmul(reshaped, weight_in) + biase_in
            # 将数据转为3维，作为LSTM_cell的输入
            X_in = tf.reshape(X_in, [-1, self.n_steps, self.n_hidden])
            # 定义一层 LSTM_cell，只需要说明 n_hidden, 它会自动匹配输入的 X 的维度
            lstm_cell = rnn.BasicLSTMCell(self.n_hidden, forget_bias=1.0, state_is_tuple=True)
            # 添加 dropout layer, 一般只设置 output_keep_prob
            if keep_prob is not None:
                lstm_cell = rnn.DropoutWrapper(cell=lstm_cell, input_keep_prob=1.0, output_keep_prob=keep_prob)
            else:
                lstm_cell = rnn.DropoutWrapper(cell=lstm_cell, input_keep_prob=1.0, output_keep_prob=self.keep_prob)
            # 调用 MultiRNNCell 来实现多层 LSTM
            lstm_cell = rnn.MultiRNNCell([lstm_cell] * self.layer_num, state_is_tuple=True)
            # 用全零来初始化state
            if batch_size is not None:
                init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
            else:
                init_state = lstm_cell.zero_state(self.batch_size, dtype=tf.float32)
            # 调用 dynamic_rnn() 来让我们构建好的网络运行起来
            # 当 time_major==False 时， outputs.shape = [batch_size, n_steps, n_hidden]
            # 所以，可以取 h_state = outputs[:, -1, :] 作为最后输出
            # state.shape = [layer_num, 2, batch_size, n_hidden],
            # 或者，可以取 h_state = state[-1][1] 作为最后输出
            # 最后输出维度是 [batch_size, n_hidden]
            outputs, finnal_state = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=init_state, time_major=False)
            last_state = tf.matmul(finnal_state[-1][1], weight_out) + biase_out
            if regularizer is not None:
                tf.add_to_collection('losses', regularizer(weight_in))
            weight_in_mean = tf.reduce_mean(weight_in)
            tf.summary.scalar('weight_in_mean', weight_in_mean)
            tf.summary.histogram('weight_in_mean', weight_in_mean)
            weight_out_mean = tf.reduce_mean(weight_out)
            tf.summary.scalar('weight_out_mean', weight_out_mean)
            tf.summary.histogram('weight_out_mean', weight_out_mean)
        return last_state

    # 定义损失函数
    def losses(self, logits, labels):
        with tf.variable_scope('loss') as scope:
            logits = tf.reshape(logits, [-1, self.class_num])
            labels = tf.reshape(labels, [-1, self.class_num])
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels)
            ses_loss = tf.reduce_mean(cross_entropy)
            tf.add_to_collection('losses', ses_loss)
            loss = tf.add_n(tf.get_collection('losses'))
            tf.summary.scalar(scope.name + '/loss', loss)
        return loss

    # 定义梯度下降函数
    def trainning(self, loss):
        with tf.name_scope('optimizer'):
            optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
            global_step = tf.Variable(0, name='global_step', trainable=False)
            train_op = optimizer.minimize(loss, global_step=global_step)
        return train_op

    # 定义模型的准确率
    def evaluation(self, logits, labels):
        with tf.variable_scope('accuracy') as scope:
            logits = tf.reshape(logits, [-1, self.class_num])
            labels = tf.reshape(labels, [-1, self.class_num])
            correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar('accuracy', accuracy)
        return accuracy

    # 训练模型
    def run_training(self, txt_dir, logs_train_dir):
        train, train_label = self.read_data(txt_dir)
        train_batch, train_label_batch = self.get_batch(train, train_label)
        # 使用L2正则化方法
        regularizer = tf.contrib.layers.l2_regularizer(self.regularaztion_rate)
        # LSTM网络
        train_logits = self.inference(train_batch, regularizer=regularizer)
        train_loss = self.losses(train_logits, train_label_batch)
        train_op = self.trainning(train_loss)
        train_acc = self.evaluation(logits=train_logits, labels=train_label_batch)
        # 合并摘要，包含所有输入摘要的值
        summary_op = tf.summary.merge_all()
        # 使用GPU训练
        config = tf.ConfigProto()
        # 占用GPU70%的显存,超出部分使用内存
        config.gpu_options.per_process_gpu_memory_fraction = 0.7
        sess = tf.Session(config=config)
        train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        # 协调器，协调线程间的关系可以视为一种信号量，用来做同步
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            step = 0
            print('start training......')
            while True:
                if coord.should_stop():
                    break
                start_time = time.time()
                step += 1
                _, tra_loss, tra_acc = sess.run([train_op, train_loss, train_acc])
                duration = time.time() - start_time
                if step % 100 == 0:
                    print('Step %d, train loss = %.5f, train acc = %.5f,  train time = %.5f' % (step, tra_loss, tra_acc, duration))
                    summary_str = sess.run(summary_op)
                    train_writer.add_summary(summary_str, step)
                if step % 500 == 0:
                    checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
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
    def check_lstm(self, data, logs_dir):
        npdata = np.reshape(data, [self.n_input, 1])
        with tf.Graph().as_default():
            X = tf.placeholder(tf.float32, shape=[self.n_input, 1])
            logit = self.inference(X, batch_size=1)
            logit = tf.nn.softmax(logit)
            saver = tf.train.Saver()
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(logs_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    global_step = ckpt.model_checkpoint_path.split('.')[-1].split('-')[-1]
                    saver.restore(sess, ckpt.model_checkpoint_path)
                else:
                    print('No checkpoint file found')
                prediction = sess.run(logit, feed_dict={X: npdata})
                outputdata = np.reshape(prediction, [-1, self.n_input, self.class_num])
                # 由于前面处理数据的时候将0去掉了，这需要把最大值再加上1
                outputdata = (outputdata[0].argmax(1) + 1).tolist()
                return outputdata

    def read_list(self, filename):
        with open(filename) as f:
            data = f.readline()
        list_data = [data.split(' ')[i:i + self.n_input] for i in range(0, len(data.split(' ')), self.n_input)]
        return list_data


if __name__ == '__main__':
    lobj = lstm(LEARNING_RATE, REGULARAZTION_RATE, BATCH_SIZE, CAPACITY, N_INPUT, N_STEPS, N_HIDDEN, LAYER_NUM, CLASS_NUM, OUTPUTNUM, KEEP_PROB)
    # lobj.run_training('E:\\aa_new.txt', 'E:\\log')
    data = lobj.read_list('E:\\aa_new.txt')
    for i in data:
        print('input_data: ', i)
        output = lobj.check_lstm(i, 'E:\\log')
        # output = [reverse_data[x] for x in output]
        print('output_data: ', output)
        time.sleep(2)
