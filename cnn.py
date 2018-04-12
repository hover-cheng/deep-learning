import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
import time

IMAGE_SIZE = 80
IMAGE_DEEP = 3
BATCH_SIZE = 200
CAPACITY = 2000 + 4 * BATCH_SIZE
REGULARAZTION_RATE = 0.001
LEARNING_RATE = 0.0001
KEEP_PROB = 0.75
CLASS_NUM = 26
OUTPUTNUM = 4

# 在tensorflow的log日志等级如下：
# - 0：显示所有日志（默认等级）
# - 1：显示info、warning和error日志
# - 2：显示warning和error信息
# - 3：显示error日志信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class cnn(object):
    def __init__(self, LEARNING_RATE, REGULARAZTION_RATE, IMAGE_SIZE, IMAGE_DEEP, BATCH_SIZE, CAPACITY, CLASS_NUM, KEEP_PROB, OUTPUTNUM):
        self.learning_rate = LEARNING_RATE
        self.regularaztion_rate = REGULARAZTION_RATE
        self.image_size = IMAGE_SIZE
        self.image_deep = IMAGE_DEEP
        self.batch_size = BATCH_SIZE
        self.capacity = CAPACITY
        self.class_num = CLASS_NUM
        self.keep_prob = KEEP_PROB
        self.outputnum = OUTPUTNUM

    # 将文件加中的文件重命名为统一格式
    def rename_image(self, filename):
        n = 0
        oldfiles = os.listdir(filename)
        newfiles = '\\'.join(filename.split('\\')[:-1])
        fileid = filename.split('\\')[-1].split('_')[0]
        for i in oldfiles:
            os.renames(filename + '\\' + i, newfiles + '\\' + str(fileid) + '_' + str(n) + '.jpg')
            n += 1

    # 生成向量以及one-hot类型的标签,图片文件名样式为: abcd.jpg
    def get_file_1(self, filepath):
        list_char = [chr(i) for i in range(97, 123)]
        dict_char = dict(enumerate(list_char))
        reversed_char = dict(zip(dict_char.values(), dict_char.keys()))
        pic_list = []
        pic_label = []
        init_label = [0 for i in range(self.class_num)]
        filename = os.listdir(filepath)
        for i in filename:
            pic_list.append(os.path.join(filepath, i))
            for j in i.split('.')[0]:
                tmp = init_label[:]
                tmp[reversed_char[j]] = 1
                pic_label.append(tmp)
        np_pic = np.array(pic_list)
        np_label = np.reshape(np.concatenate(np.array(pic_label, np.float32)), [-1, self.class_num * self.outputnum])
        return np_pic, np_label

    # 生成向量以及one-hot类型的标签,图片文件名样式为: 0_286.jpg
    def get_file(self, filepath):
        pic_list = []
        pic_label = []
        # 分五个类别
        # 0:向日葵 1:蒲公英 2:雏菊 3:玫瑰 4:郁金香
        init_label = [0 for i in range(self.class_num)]
        filename = os.listdir(filepath)
        for i in filename:
            pic_list.append(os.path.join(filepath, i))
            real_label = init_label[:]
            real_label[int(i.split('_')[0])] = 1
            pic_label.append(real_label)
        return pic_list, pic_label

    def get_batch(self, image, label):
        input_queue = tf.train.slice_input_producer([image, label])
        label = input_queue[1]
        image_contents = tf.read_file(input_queue[0])
        image = tf.image.decode_jpeg(image_contents, channels=3)
        # 将图片尺寸调整
        # 其中 method 有四种选择：
        # ResizeMethod.BILINEAR ：双线性插值
        # ResizeMethod.NEAREST_NEIGHBOR ： 最近邻插值
        # ResizeMethod.BICUBIC ： 双三次插值
        # ResizeMethod.AREA ：面积插值
        image = tf.image.resize_images(image, [self.image_size, self.image_size], method=1)
        # # 将图片转换为灰度图片
        # image = tf.image.rgb_to_grayscale(image)
        # image = tf.image.convert_image_dtype(image, tf.uint8)
        # 对图像进行标准化，将图片的像素数据限定到一个范围，加速神经网络的训练
        image = tf.image.per_image_standardization(image)
        # 对图像进行标准化，将数据转化到0-1之间，因为图像数据均为0-255之间的数据
        # image = tf.cast(image, tf.float32) / 255.
        # 生成batch
        pic_batch, label_batch = tf.train.shuffle_batch([image, label], batch_size=self.batch_size, num_threads=4, capacity=self.capacity, min_after_dequeue=2 * self.batch_size)
        # label_batch = tf.cast(label_batch, tf.float32)
        # 由于是5分类，故将label数据的shape转化为(batch_size, 5)
        label_batch = tf.reshape(label_batch, [-1, self.class_num * self.outputnum])
        return pic_batch, label_batch

    # 定义卷积层函数
    def conv_net(self, name, inputdata, w_shape, strides_shape, padding):
        with tf.variable_scope(name):
            weight_name = name + '_weight'
            biases_name = name + '_biases'
            conve_name = name + '_conve'
            bias_name = name + '_bias'
            actived_name = name + '_actived'
            weight_name = tf.get_variable('weights', w_shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
            biases_name = tf.get_variable('biases', w_shape[-1], initializer=tf.constant_initializer(0.0))
            conve_name = tf.nn.conv2d(inputdata, weight_name, strides=strides_shape, padding=padding)
            bias_name = tf.nn.bias_add(conve_name, biases_name)
            # bn = tf.layers.batch_normalization(bias_name, trainable=YORN)
            actived_name = tf.nn.relu(bias_name)
        return actived_name

    # 定义最大池化层函数
    def max_pool(self, name, inputdata, kszie, strides, padding):
        with tf.variable_scope(name):
            pool_name = name + '_pool'
            pool_name = tf.nn.max_pool(inputdata, ksize=kszie, strides=strides, padding=padding)
        return pool_name

    # 定义平均池化层函数
    def avg_pool(self, name, inputdata, kszie, strides, padding):
        with tf.variable_scope(name):
            pool_name = name + '_pool'
            pool_name = tf.nn.avg_pool(inputdata, ksize=kszie, strides=strides, padding=padding)
        return pool_name

    # 定义局部标准化层函数
    def norm_net(self, name, inputdata, lsize=4):
        with tf.variable_scope(name):
            norm = tf.nn.lrn(inputdata, lsize, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)
        return norm

    # 定义全链接层
    def fc_net(self, name, inputdata, w_shape, regularizer):
        with tf.variable_scope(name):
            weight_name = name + '_weight'
            biases_name = name + '_biases'
            conve_name = name + '_conve'
            mean_name = name + '_mean'
            stddev_name = name + '_stddev'
            weight_name = tf.get_variable('weights', w_shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
            if regularizer is not None:
                tf.add_to_collection('losses', regularizer(weight_name))
            biases_name = tf.get_variable('biases', w_shape[-1], initializer=tf.constant_initializer(0.1))
            conve_name = tf.matmul(inputdata, weight_name) + biases_name
            # 生成可视化图
            tf.summary.histogram(name + '_weight', weight_name)
            mean_name = tf.reduce_mean(weight_name)
            tf.summary.scalar(name + '_mean', mean_name)
            stddev_name = tf.sqrt(tf.reduce_mean(tf.square(weight_name - mean_name)))
            tf.summary.scalar(name + '_stddev', stddev_name)
        return conve_name

    # 定义激活函数层
    def relu_net(self, inputdata):
        return tf.nn.relu(inputdata)

    # 定义dropout层
    def drop_net(self, inputdata, keep_prob):
        if keep_prob is not None:
            return tf.nn.dropout(inputdata, keep_prob=keep_prob)
        else:
            return tf.nn.dropout(inputdata, keep_prob=self.keep_prob)

    # 使用卷积神经网络
    def inference(self, image, keep_prob=None, regularizer=None):
        actived_conv1 = self.conv_net('conv1', image, [3, 3, 3, 64], [1, 1, 1, 1], 'SAME')
        pool1 = self.max_pool('pool1', actived_conv1, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
        actived_conv2 = self.conv_net('conv2', pool1, [3, 3, 64, 128], [1, 1, 1, 1], 'SAME')
        pool2 = self.max_pool('pool2', actived_conv2, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
        actived_conv3 = self.conv_net('conv3', pool2, [3, 3, 128, 256], [1, 1, 1, 1], 'SAME')
        pool3 = self.max_pool('pool3', actived_conv3, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
        actived_conv4 = self.conv_net('conv4', pool3, [3, 3, 256, 512], [1, 1, 1, 1], 'SAME')
        pool4 = self.max_pool('pool4', actived_conv4, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
        pool_shape = pool4.get_shape().as_list()
        nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
        reshaped = tf.reshape(pool4, [pool_shape[0], nodes])
        fc1 = self.fc_net('fc1', reshaped, [nodes, 256], regularizer)
        fc1_relu = self.relu_net(fc1)
        fc1_drop = self.drop_net(fc1_relu, keep_prob)
        fc2 = self.fc_net('fc2', fc1_drop, [256, 128], regularizer)
        fc2_relu = self.relu_net(fc2)
        fc2_drop = self.drop_net(fc2_relu, keep_prob)
        fc3 = self.fc_net('fc3', fc2_drop, [128, self.class_num * self.outputnum], regularizer)
        return fc3

    # 损失函数
    def losses(self, logits, labels):
        with tf.variable_scope('loss') as scope:
            logits = tf.reshape(logits, [-1, self.class_num])
            labels = tf.reshape(labels, [-1, self.class_num])
            # 计算logits和label之间的交叉熵，并且直接给出softmax回归结果，其中要求label输入的格式是：类别，0,1,2,3,4,5,6,7,8,9（10个类的话）
            # cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
            # cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
            # 计算logits和label之间的交叉熵，并给出softmax预测,其中要求label的格式必须是 one-hot类型的变量,[0,0,0,1],[0,0,1,0],[0,1,0,0],[1,0,0,0]（4个类别的话）
            # softmax函数返回的向量值的加和是1，例如[[0.005, 0.005, 0.030, 0.620, 0.250], [1., 0., 0., 0., 0.], [0.144, 0.232, 0.062, 0.326, 0.236]]
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels)
            # 均方差损失函数
            # cross_entropy = tf.square(logits - labels)
            # 最小二乘法损失函数
            # cross_entropy = tf.pow(labels - logits, 2)
            # # 获取名称为'losses'的集合中的元素，并且求和
            ses_loss = tf.reduce_mean(cross_entropy)
            tf.add_to_collection('losses', ses_loss)
            loss = tf.add_n(tf.get_collection('losses'))
            tf.summary.scalar(scope.name + '/loss', loss)
        return loss

    # 训练函数
    def trainning(self, loss):
        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdadeltaOptimizer(learning_rate=self.learning_rate)
            global_step = tf.Variable(0, name='global_step', trainable=False)
            train_op = optimizer.minimize(loss, global_step=global_step)
        return train_op

    # 定义模型的准确率
    def evaluation(self, logits, labels):
        with tf.variable_scope('accuracy') as scope:
            # 将结果转换
            logits = tf.reshape(logits, [-1, self.class_num])
            labels = tf.reshape(labels, [-1, self.class_num])
            # 分别取出logits和labels的行最大值的位置，并且比较是否相等
            correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
            # correct_prediction = tf.equal(labels, logits)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar('accuracy', accuracy)
        return accuracy

    # 图片分类训练(CNN)
    def run_training(self, train_dir, logs_train_dir):
        # 运行 tensorboard --logdir=E:\\log，然后访问 http://127.0.0.1:6006/ 可以查看到可视的训练数据
        # tensorboard 程序不支持中文路径，并且最好放到根目录
        train, train_label = self.get_file_1(train_dir)
        print('the sum of the training pictures is: %d' % len(train))
        train_batch, train_label_batch = self.get_batch(train, train_label)
        # 使用L2正则化方法
        regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
        # 卷积网络
        train_logits = self.inference(train_batch, keep_prob=self.keep_prob, regularizer=regularizer)
        train_loss = self.losses(train_logits, train_label_batch)
        train_op = self.trainning(train_loss)
        train_acc = self.evaluation(train_logits, train_label_batch)
        # 添加包含图片的摘要
        image_op = tf.summary.image('input', train_batch, self.batch_size)
        # image_op_1 = tf.summary.image('output', train_logits, BATCH_SIZE)
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
                step += 1
                start_time = time.time()
                _, tra_loss, tra_acc = sess.run([train_op, train_loss, train_acc])
                duration = time.time() - start_time
                if step % 100 == 0:
                    print('Step %d, train loss = %.5f, train acc = %.5f, train time = %.5f' % (step, tra_loss, tra_acc, duration))
                    # print('Step %d, train loss = %.5f, train time = %.5f' % (step, tra_loss, duration))
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

    # 图片分类验证
    def check_train(self, test_dir, logs_dir):
        filesname = os.listdir(test_dir)
        image_arry = []
        image_label = []
        list_char = [chr(i) for i in range(97, 123)]
        dict_char = dict(enumerate(list_char))
        reversed_char = dict(zip(dict_char.values(), dict_char.keys()))
        count = 0
        init_label = [0 for i in range(self.class_num)]
        print('the sum of the test pictures is: %d' % len(filesname))
        for i in filesname:
            image = Image.open(os.path.join(test_dir, i))
            # 模式“I”为32位整型灰色图像，它的每个像素用32个bit表示，0表示黑，255表示白，(0,255)之间的数字表示不同的灰度
            # 模式L”为灰色图像，它的每个像素用8个bit表示，0表示黑，255表示白，其他数字表示不同的灰度
            # image_L = image.convert('RGBA')
            image_L = image.resize([self.image_size, self.image_size])
            image_L = np.array(image_L)
            image_arry.append(image_L)
            for j in i.split('.')[0]:
                tmp = init_label[:]
                tmp[reversed_char[j]] = 1
                image_label.append(tmp)
        np_label = np.reshape(np.concatenate(np.array(image_label, np.float32)), [-1, self.outputnum, self.class_num])
        with tf.Graph().as_default():
            X = tf.placeholder(tf.float32, shape=[1, self.image_size, self.image_size, self.image_deep])
            logit = self.inference(X, keep_prob=1)
            logit = tf.nn.softmax(logit)
            saver = tf.train.Saver()
            with tf.Session() as sess:
                # print('Reading checkpoints...')
                ckpt = tf.train.get_checkpoint_state(logs_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    # print(ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('.')[-1].split('-')[-1]
                    saver.restore(sess, ckpt.model_checkpoint_path)
                else:
                    print('No checkpoint file found')
                for i in range(len(image_arry)):
                    image = np.reshape(image_arry[i], [1, self.image_size, self.image_size, self.image_deep])
                    prediction = sess.run(logit, feed_dict={X: image})
                    outputdata = np.reshape(prediction, [self.outputnum, self.class_num])
                    test_label = outputdata.argmax(1).tolist()
                    real_label = np_label[i].argmax(1).tolist()
                    if real_label == test_label:
                            count += 1
                    else:
                        print('real_label:', real_label, 'test_label:', test_label)
                acc = float(count) / float(len(image_arry))
                print('acc is %0.5f' % acc)


if __name__ == '__main__':
    lobj = cnn(LEARNING_RATE, REGULARAZTION_RATE, IMAGE_SIZE, IMAGE_DEEP, BATCH_SIZE, CAPACITY, CLASS_NUM, KEEP_PROB, OUTPUTNUM)
    lobj.run_training('train', 'E:\\log')
