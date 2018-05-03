import tensorflow as tf
import numpy as np
import os
from PIL import Image
import time

IMAGE_SIZE = 300
IMAGE_DEEP = 3
BATCH_SIZE = 100
CAPACITY = 2000 + 4 * BATCH_SIZE
REGULARAZTION_RATE = 0.001
LEARNING_RATE = 0.0001
KEEP_PROB = 0.5
CLASS_NUM = 5
OUTPUTNUM = 1

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

'''
{'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7, 'i': 8, 'j': 9, 'k': 10, 'l': 11, 'm': 12, 'n': 13, 'o': 14, 'p': 15, 'q': 16, 'r': 17, 's': 18, 't': 19, 'u': 20, 'v': 21, 'w': 22, 'x': 23, 'y'
: 24, 'z': 25}
'''


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

    def rename_image(self, filename):
        n = 0
        oldfiles = os.listdir(filename)
        newfiles = '\\'.join(filename.split('\\')[:-1])
        fileid = filename.split('\\')[-1].split('_')[0]
        for i in oldfiles:
            os.renames(filename + '\\' + i, newfiles + '\\' + str(fileid) + '_' + str(n) + '.jpg')
            n += 1

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

    def get_file(self, filepath):
        pic_list = []
        pic_label = []
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
        image = tf.image.resize_images(image, [self.image_size, self.image_size], method=1)
        image = tf.cast(image, tf.float32) / 255.
        pic_batch, label_batch = tf.train.shuffle_batch([image, label], batch_size=self.batch_size, num_threads=4, capacity=self.capacity, min_after_dequeue=2 * self.batch_size)
        label_batch = tf.reshape(label_batch, [-1, self.class_num * self.outputnum])
        return pic_batch, label_batch

    def conv_net(self, name, inputdata, w_shape, strides_shape, padding):
        with tf.variable_scope(name):
            weight_name = name + '_weight'
            biases_name = name + '_biases'
            conve_name = name + '_conve'
            bias_name = name + '_bias'
            actived_name = name + '_actived'
            weight_name = tf.get_variable('weights', w_shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
            biases_name = tf.get_variable('biases', w_shape[-1], initializer=tf.constant_initializer(0.1))
            conve_name = tf.nn.conv2d(inputdata, weight_name, strides=strides_shape, padding=padding)
            bias_name = tf.nn.bias_add(conve_name, biases_name)
            actived_name = tf.nn.relu(bias_name)
        return actived_name

    def max_pool(self, name, inputdata, kszie, strides, padding):
        with tf.variable_scope(name):
            pool_name = name + '_pool'
            pool_name = tf.nn.max_pool(inputdata, ksize=kszie, strides=strides, padding=padding)
        return pool_name

    def avg_pool(self, name, inputdata, kszie, strides, padding):
        with tf.variable_scope(name):
            pool_name = name + '_pool'
            pool_name = tf.nn.avg_pool(inputdata, ksize=kszie, strides=strides, padding=padding)
        return pool_name

    def norm_net(self, name, inputdata, lsize=4):
        with tf.variable_scope(name):
            norm = tf.nn.lrn(inputdata, lsize, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)
        return norm

    def fc_net(self, name, inputdata, w_shape, regularizer=None):
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
            tf.summary.histogram(name + '_weight', weight_name)
            mean_name = tf.reduce_mean(weight_name)
            tf.summary.scalar(name + '_mean', mean_name)
            stddev_name = tf.sqrt(tf.reduce_mean(tf.square(weight_name - mean_name)))
            tf.summary.scalar(name + '_stddev', stddev_name)
        return conve_name

    def relu_net(self, inputdata):
        return tf.nn.relu(inputdata)

    def drop_net(self, inputdata, keep_prob=None):
        if keep_prob is not None:
            return tf.nn.dropout(inputdata, keep_prob=keep_prob)
        else:
            return tf.nn.dropout(inputdata, keep_prob=self.keep_prob)

    def inference(self, image, keep_prob=None, regularizer=None):
        actived_conv1 = self.conv_net('conv1', image, [11, 11, 3, 64], [1, 4, 4, 1], 'SAME')
        pool1 = self.max_pool('pool1', actived_conv1, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME')
        actived_conv2 = self.conv_net('conv2', pool1, [5, 5, 64, 192], [1, 1, 1, 1], 'SAME')
        pool2 = self.max_pool('pool2', actived_conv2, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME')
        actived_conv3 = self.conv_net('conv3', pool2, [3, 3, 192, 384], [1, 1, 1, 1], 'SAME')
        actived_conv4 = self.conv_net('conv4', actived_conv3, [3, 3, 384, 256], [1, 1, 1, 1], 'SAME')
        pool3 = self.max_pool('pool3', actived_conv4, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME')
        actived_conv5 = self.conv_net('conv5', pool3, [3, 3, 256, 256], [1, 1, 1, 1], 'SAME')
        pool4 = self.max_pool('pool4', actived_conv5, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
        pool_shape = pool4.get_shape().as_list()
        nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
        reshaped = tf.reshape(pool4, [pool_shape[0], nodes])
        reshaped_drop = self.drop_net(reshaped, keep_prob)
        fc1 = self.fc_net('fc1', reshaped_drop, [nodes, 1000], regularizer)
        fc1_relu = self.relu_net(fc1)
        fc1_drop = self.drop_net(fc1_relu, keep_prob)
        fc2 = self.fc_net('fc2', fc1_drop, [1000, self.class_num * self.outputnum], regularizer)
        return fc2

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

    def trainning(self, loss):
        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            global_step = tf.Variable(0, name='global_step', trainable=False)
            train_op = optimizer.minimize(loss, global_step=global_step)
        return train_op

    def evaluation(self, logits, labels):
        with tf.variable_scope('accuracy') as scope:
            logits = tf.reshape(logits, [-1, self.class_num])
            labels = tf.reshape(labels, [-1, self.class_num])
            correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar('accuracy', accuracy)
        return accuracy

    def run_training(self, train_dir, logs_train_dir):
        with tf.Graph().as_default():
            train, train_label = self.get_file(train_dir)
            print('the sum of the training pictures is: %d' % len(train))
            train_batch, train_label_batch = self.get_batch(train, train_label)
            regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
            train_logits = self.inference(train_batch, keep_prob=self.keep_prob, regularizer=regularizer)
            train_loss = self.losses(train_logits, train_label_batch)
            train_op = self.trainning(train_loss)
            train_acc = self.evaluation(train_logits, train_label_batch)
            image_op = tf.summary.image('input', train_batch, self.batch_size)
            summary_op = tf.summary.merge_all()
            config = tf.ConfigProto()
            config.gpu_options.per_process_gpu_memory_fraction = 0.8
            sess = tf.Session(config=config)
            train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())
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
                    # aaa, bbb = sess.run([train_batch, train_label_batch])
                    # bbb = bbb.reshape(-1, 4, 26)
                    # print(aaa[2].shape)
                    # plt.imshow(aaa[2])
                    # plt.show()
                    # print(bbb[2].argmax(1))
                    # break
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
                coord.request_stop()
            coord.join(threads)
            sess.close()

    def check_train(self, test_dir, logs_dir):
        filesname = os.listdir(test_dir)
        image_arry = []
        image_label = []
        n = 5
        a = 0
        test_count = [0 for i in range(n)]
        real_count = [0 for i in range(n)]
        each_count = [0 for i in range(n)]
        init_label = [0 for i in range(n)]
        # list_char = [chr(i) for i in range(97, 123)]
        # dict_char = dict(enumerate(list_char))
        # reversed_char = dict(zip(dict_char.values(), dict_char.keys()))
        # count = 0
        # init_label = [0 for i in range(self.class_num)]
        print('the sum of the test pictures is: %d' % len(filesname))
        for i in filesname:
            image = Image.open(os.path.join(test_dir, i))
            # image_L = image.convert('RGBA')
            image_L = image.resize([self.image_size, self.image_size])
            image_L = np.array(image_L)
            image_arry.append(image_L)
            # for j in i.split('.')[0]:
            #     tmp = init_label[:]
            #     tmp[reversed_char[j]] = 1
            #     image_label.append(tmp)
        # np_label = np.reshape(np.concatenate(np.array(image_label, np.float32)), [-1, self.outputnum, self.class_num])
            real_label = init_label[:]
            real_label[int(i.split('_')[0])] = 1
            real_count[int(i.split('_')[0])] += 1
            image_label.append(real_label)
        np_label = np.array(image_label, dtype=np.float32)
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
                    # outputdata = np.reshape(prediction, [self.outputnum, self.class_num])
                    # test_label = outputdata.argmax(1).tolist()
                    # real_label = np_label[i].argmax(1).tolist()
                    #  if real_label == test_label:
                    #         count += 1
                    # else:
                    #     print('real_label:', real_label, 'test_label:', test_label)
                # acc = float(count) / float(len(image_arry))
                # print('acc is %0.5f' % acc)
                    test_label = np.argmax(prediction)
                    real_label = np.argmax(np_label[i])
                    test_count[test_label] += 1
                    if real_label == test_label:
                        a += 1
                        each_count[real_label] += 1
                    else:
                        print(test_label, real_label, prediction, filesname[i])
                acc = float(a) / float(len(image_arry))
                print('acc is %0.5f' % acc)
                print('the number of each prediction category is %s' % test_count)
                print('the number of each real category is %s' % real_count)
                print('the number of accurate prediction category is %s' % each_count)


    def application(self, app_dir, logs_dir):
        filesname = os.listdir(app_dir)
        image_arry = []
        typename = {0: 'Sunflower', 1: 'Dandelion', 2: 'Daisies', 3: 'Rose', 4: 'Tulips'}
        # print('the sum of the test pictures is: %d' % len(filesname))
        for i in filesname:
            image = Image.open(os.path.join(app_dir, i))
            image_L = image.resize([self.image_size, self.image_size])
            image_L = np.array(image_L)
            image_arry.append(image_L)
        with tf.Graph().as_default():
            X = tf.placeholder(tf.float32, shape=[1, self.image_size, self.image_size, self.image_deep])
            logit = self.inference(X, keep_prob=1)
            logit = tf.nn.softmax(logit)
            saver = tf.train.Saver()
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(logs_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    global_step = ckpt.model_checkpoint_path.split('.')[-1].split('-')[-1]
                    saver.restore(sess, ckpt.model_checkpoint_path)
                else:
                    print('No checkpoint file found')
                for i in range(len(image_arry)):
                    image = np.reshape(image_arry[i], [1, self.image_size, self.image_size, self.image_deep])
                    prediction = sess.run(logit, feed_dict={X: image})
                    print(filesname[i], 'is', typename[int(prediction.argmax())])

if __name__ == '__main__':
    lobj = cnn(LEARNING_RATE, REGULARAZTION_RATE, IMAGE_SIZE, IMAGE_DEEP, BATCH_SIZE, CAPACITY, CLASS_NUM, KEEP_PROB, OUTPUTNUM)
    # lobj.run_training('train', '/data/log/')
    # lobj.check_train('test', '/data/log/')
    lobj.application('application', '/data/log/')
