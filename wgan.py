import tensorflow as tf
import numpy as np
import os
from tensorflow.examples.tutorials.mnist import input_data
# pip install scikit-image
from skimage.io import imsave
from skimage import io
from PIL import Image
import matplotlib.pyplot as plt
import time

img_height = 200
img_width = 200
img_channel = 3
img_size = img_height * img_width * img_channel
output_path = "output"
max_epoch = 500000
h1_size = 200
h2_size = 300
h3_size = 400
z_size = 100
batch_size = 32
capacity = 2000 + 4 * batch_size
learning_rate = 0.0001
keep_prob = 0.7
CLIP = [-0.01, 0.01]
CRITIC_NUM = 5


# 生成向量以及one-hot类型的标签,图片文件名样式为: 0_286.jpg
def get_file(filepath):
    pic_list = []
    # 分五个类别
    # 0:向日葵 1:蒲公英 2:雏菊 3:玫瑰 4:郁金香
    filename = os.listdir(filepath)
    for i in filename:
        pic_list.append(os.path.join(filepath, i))
    return pic_list


def get_batch(image):
    input_queue = tf.train.slice_input_producer([image])
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents, channels=3)
    # 将图片尺寸调整
    # 其中 method 有四种选择：
    # ResizeMethod.BILINEAR ：双线性插值
    # ResizeMethod.NEAREST_NEIGHBOR ： 最近邻插值
    # ResizeMethod.BICUBIC ： 双三次插值
    # ResizeMethod.AREA ：面积插值
    image = tf.image.resize_images(image, [img_height, img_width], method=1)
    # # 将图片转换为灰度图片
    # image = tf.image.rgb_to_grayscale(image)
    # image = tf.image.convert_image_dtype(image, tf.uint8)
    # 对图像进行标准化，将图片的像素数据限定到一个范围，加速神经网络的训练
    # image = tf.image.per_image_standardization(image)
    # 对图像进行标准化，将数据转化到0-1之间，因为图像数据均为0-255之间的数据
    image = tf.cast(image, tf.float32) / 255.
    # 生成batch
    pic_batch = tf.train.shuffle_batch([image], batch_size=batch_size, num_threads=4, capacity=capacity, min_after_dequeue=2 * batch_size)
    pic_batch = tf.reshape(pic_batch, [-1, img_size])
    # pic_batch = 2 * pic_batch - 1
    # label_batch = tf.cast(label_batch, tf.float32)
    # 由于是5分类，故将label数据的shape转化为(batch_size, 5)
    return pic_batch


def conv_net(name, inputdata, w_shape, strides_shape, padding='SAME'):
    with tf.variable_scope(name):
        weight = tf.get_variable('w', w_shape, initializer=tf.truncated_normal_initializer(stddev=0.02))
        biases = tf.get_variable('b', w_shape[-1], initializer=tf.constant_initializer(0))
        conve = tf.nn.conv2d(inputdata, weight, strides=strides_shape, padding=padding)
        bias = tf.nn.bias_add(conve, biases)
        return bias


# 定义反卷积层
# tf.nn.conv2d_transpose(value, filter, output_shape, strides, padding='SAME', name=None)
# value是上一层的feature map
# filter是卷积核[kernel_size, kernel_size, output_channel, input_channel ]
# output_shape定义输出的尺寸[batch_size, height, width, channel]
# padding是边界打补丁的算法
def deconv_net(name, inputdata, w_shape, strides_shape, padding='SAME', output_shape=None):
    with tf.variable_scope(name):
        input_dim = inputdata.get_shape()[-1]
        input_height = int(inputdata.get_shape()[1])
        input_width = int(inputdata.get_shape()[2])
        if output_shape:
            output_shape = output_shape
        else:
            output_shape = [batch_size, input_height * 2, input_width * 2, w_shape[-2]]
        weight = tf.get_variable('w', w_shape, initializer=tf.truncated_normal_initializer(stddev=0.02))
        biases = tf.get_variable('b', w_shape[-2], initializer=tf.constant_initializer(0))
        conve = tf.nn.conv2d_transpose(inputdata, weight, output_shape, strides=strides_shape, padding=padding)
        bias = tf.nn.bias_add(conve, biases)
    return bias


# 定义lrelu激活层
def lrelu(x, leak=0.02, name="lrelu"):
    return tf.maximum(leak * x, x)


# 定义最大池化层函数
def max_pool(name, inputdata, kszie, strides, padding):
    with tf.variable_scope(name):
        pool_name = tf.nn.max_pool(inputdata, ksize=kszie, strides=strides, padding=padding)
    return pool_name


# 定义dropout层
def drop_net(inputdata, keep_prob=None):
    if keep_prob is not None:
        return tf.nn.dropout(inputdata, keep_prob=keep_prob)
    else:
        return tf.nn.dropout(inputdata, keep_prob=1.)


def batch_norm(inputdata, name="batch_norm", scope="scope", reuse=False):
    with tf.variable_scope(name):
        return tf.contrib.layers.batch_norm(inputdata, epsilon=1e-5, decay=0.9, scale=True, scope=scope, reuse=reuse, updates_collections=None)


def generator(name, z, reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        g_w1 = tf.Variable(tf.truncated_normal([z_size, 25 * 25 * 256], stddev=0.02), name="g_w1", dtype=tf.float32)
        b1 = tf.Variable(tf.zeros([25 * 25 * 256]), name="g_b1", dtype=tf.float32)
        h1 = tf.matmul(z, g_w1) + b1
        h1 = tf.nn.relu(batch_norm(h1, name='g_bn1'))
        h1 = tf.reshape(h1, [batch_size, 25, 25, 256])
        deconv_2 = deconv_net(inputdata=h1, w_shape=[3, 3, 128, 256], strides_shape=[1, 2, 2, 1], name='g_d2')
        deconv_2 = tf.nn.relu(batch_norm(deconv_2, name='g_bn2'))
        # # [batch_size, 7, 7, 128]
        deconv_3 = deconv_net(inputdata=deconv_2, w_shape=[3, 3, 64, 128], strides_shape=[1, 2, 2, 1], name='g_d3')
        deconv_3 = tf.nn.relu(batch_norm(deconv_3, name='g_bn3'))
        # # # [batch_size, 14, 14, 64]
        deconv_4 = deconv_net(inputdata=deconv_3, w_shape=[3, 3, img_channel, 64], strides_shape=[1, 2, 2, 1], name='g_d4')
        deconv_4 = tf.nn.relu(batch_norm(deconv_4, name='g_bn4'))
        # [batch_size, 28, 28, img_channel]
        return tf.nn.tanh(deconv_4)


def discriminator(name, x, y, reuse=False, keep_prob=1):
    with tf.variable_scope(name, reuse=reuse):
        x = tf.reshape(x, [batch_size, img_height, img_width, img_channel])
        # y = tf.reshape(y, [batch_size, img_height, img_width, img_channel])
        # 将x,y两个数据拼接为一个数据进行计算
        x = tf.concat([x, y], axis=0)
        h0 = conv_net(inputdata=x, w_shape=[3, 3, img_channel, 64], strides_shape=[1, 2, 2, 1], name='d_h0_conv')
        h0 = lrelu(batch_norm(h0, name='d_bn0'))
        h1 = conv_net(inputdata=h0, w_shape=[3, 3, 64, 128], strides_shape=[1, 2, 2, 1], name='d_h1_conv')
        h1 = lrelu(batch_norm(h1, name='d_bn1'))
        h2 = conv_net(inputdata=h1, w_shape=[3, 3, 128, 256], strides_shape=[1, 2, 2, 1], name='d_h2_conv')
        h2 = lrelu(batch_norm(h2, name='d_bn2'))
        h3 = conv_net(inputdata=h2, w_shape=[3, 3, 256, 256], strides_shape=[1, 2, 2, 1], name='d_h3_conv')
        h3 = lrelu(batch_norm(h3, name='d_bn3'))
        s1 = tf.slice(h3, [0, 0, 0, 0], [batch_size, -1, -1, -1])
        s2 = tf.slice(h3, [batch_size, 0, 0, 0], [-1, -1, -1, -1])
        h4 = tf.reshape(s1, [batch_size, -1])
        shaped = h4.get_shape().as_list()
        d_w1 = tf.Variable(tf.truncated_normal([shaped[1], 1], stddev=0.02), name="d_w1", dtype=tf.float32)
        b1 = tf.Variable(tf.zeros([1]), name="d_b1", dtype=tf.float32)
        y_data = tf.matmul(h4, d_w1) + b1
        h_f4 = tf.reshape(s2, [batch_size, -1])
        y_generated = tf.matmul(h_f4, d_w1) + b1
        return y_data, y_generated


def losses(y_data, y_generated):
    with tf.variable_scope('loss') as scope:
        d_loss = tf.reduce_mean(y_generated) - tf.reduce_mean(y_data)
        g_loss = - tf.reduce_mean(y_generated)
        return d_loss, g_loss


def training(d_loss, g_loss, learning_rate):
    with tf.name_scope('optimizer'):
        d_vars = [v for v in tf.trainable_variables() if 'd_' in v.name]
        g_vars = [v for v in tf.trainable_variables() if 'g_' in v.name]
        d_trainer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(d_loss, var_list=d_vars)
        g_trainer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(g_loss, var_list=g_vars)
        clip_d_op = [var.assign(tf.clip_by_value(var, CLIP[0], CLIP[1])) for var in d_vars]
        clip_d_op = tf.group(*clip_d_op)
        return d_trainer, g_trainer, clip_d_op


def run_training(train_dir, logs_train_dir):
    # 删除目录下的文件
    files = os.listdir(logs_train_dir)
    for i in files:
        if os.path.isfile(os.path.join(logs_train_dir, i)):
            os.remove(os.path.join(logs_train_dir, i))
    with tf.Graph().as_default():
        train = get_file(train_dir)
        train_batch = get_batch(train)
        z_prior = tf.placeholder(tf.float32, [batch_size, z_size], name="z_prior")
        x_generated = generator('gen', z_prior)
        y_data, y_generated = discriminator('dis_r', train_batch, x_generated, keep_prob=keep_prob)
        d_loss, g_loss = losses(y_data, y_generated)
        d_trainer, g_trainer, clip_d_ops = training(d_loss, g_loss, learning_rate)
        train_img = tf.reshape(train_batch, [batch_size, img_height, img_width, img_channel])
        out_img = tf.reshape(x_generated, [batch_size, img_height, img_width, img_channel])
        img_scalar = tf.summary.image('input_img', train_img, batch_size)
        out_scalar = tf.summary.image('out_img', out_img, batch_size)
        d_loss_scalar = tf.summary.scalar('loss/d_loss', d_loss)
        g_loss_scalar = tf.summary.scalar('loss/g_loss', g_loss)
        config = tf.ConfigProto()
        # 允许显存增长。如果设置为 True，分配器不会预先分配一定量 GPU 显存，而是先分配一小块，必要时增加显存分配
        config.gpu_options.allow_growth = True
        # 占用GPU70%的显存,超出部分使用内存
        config.gpu_options.per_process_gpu_memory_fraction = 0.8
        sess = tf.Session(config=config)
        train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        # 协调器，协调线程间的关系可以视为一种信号量，用来做同步
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            print('start training......')
            global_step = 1
            for step in range(global_step, max_epoch):
                start_time = time.time()
                # 每一步迭代，我们都会加载256个训练样本，然后执行一次train_step
                # # 执行判别
                if step < 25 or step % 500 == 0:
                    critic_num = 100
                else:
                    critic_num = CRITIC_NUM
                for _ in range(critic_num):
                    z_value = np.random.normal(size=(batch_size, z_size))
                    _, d_losses, _ = sess.run([d_trainer, d_loss, clip_d_ops], feed_dict={z_prior: z_value})
                z_value = np.random.normal(-1, 1, size=(batch_size, z_size))
                _, g_losses = sess.run([g_trainer, g_loss], feed_dict={z_prior: z_value})
                duration = time.time() - start_time
                if step % 10 == 0:
                    print('Step %d, G loss = %.8f, G loss=%.8f, train time = %.5f' % (step, d_losses, g_losses, duration))
                    z_value = np.random.normal(-1, 1, size=(batch_size, z_size))
                    input_met, out_met, d_loss_met, g_loss_met = sess.run([img_scalar, out_scalar, d_loss_scalar, g_loss_scalar], feed_dict={z_prior: z_value})
                    train_writer.add_summary(input_met, global_step=step)
                    train_writer.add_summary(out_met, global_step=step)
                    train_writer.add_summary(d_loss_met, global_step=step)
                    train_writer.add_summary(g_loss_met, global_step=step)
                if step % 500 == 0 or step == max_epoch:
                    z_sample_val = np.random.normal(-1, 1, size=(batch_size, z_size))
                    x_gen_val = sess.run(x_generated, feed_dict={z_prior: z_sample_val})
                    show_result(x_gen_val, os.path.join(output_path, "sample{0}.jpg".format(step)))
                    z_random_sample_val = np.random.normal(-1, 1, size=(batch_size, z_size))
                    x_gen_val = sess.run(x_generated, feed_dict={z_prior: z_random_sample_val})
                    show_result(x_gen_val, os.path.join(output_path, "random_sample{0}.jpg".format(step)))
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


def show_result(batch_res, fname, grid_size=(4, 4), grid_pad=5):
    batch_res = batch_res.reshape((batch_res.shape[0], img_height, img_width, img_channel))
    img_h, img_w = batch_res.shape[1], batch_res.shape[2]
    grid_h = img_h * grid_size[0] + grid_pad * (grid_size[0] - 1)
    grid_w = img_w * grid_size[1] + grid_pad * (grid_size[1] - 1)
    img_grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
    for i, res in enumerate(batch_res):
        if i >= grid_size[0] * grid_size[1]:
            break
        img = (res) * 255.
        img = img.astype(np.uint8)
        row = (i // grid_size[0]) * (img_h + grid_pad)
        col = (i % grid_size[1]) * (img_w + grid_pad)
        img_grid[row:row + img_h, col:col + img_w] = img
    imsave(fname, img_grid)


def check_gan(logs_dir):
    global batch_size
    batch_size = 1
    with tf.Graph().as_default():
        z_prior = tf.placeholder(tf.float32, [1, z_size], name="z_prior")
        x_generated = generator('gen', z_prior)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(logs_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('.')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                print('No checkpoint file found')
            z_test_value = np.random.normal(-1, 1, size=(1, z_size)).astype(np.float32)
            x_gen_val = sess.run(x_generated, feed_dict={z_prior: z_test_value})
            img_l = x_gen_val.reshape([img_height, img_width, img_channel])
            im = Image.fromarray(np.uint8(img_l * 255))
            plt.imshow(im)
            plt.show()
            im.save(os.path.join(output_path, "random_test.jpg"))


if __name__ == '__main__':
    run_training('vanGogh', 'log')
