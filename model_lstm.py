import tensorflow as tf
from tensorflow.contrib import rnn


# 定义LSTM网络模型
def lstm_net(name, data, n_hidden, n_steps, batch_size, regularizer=False, is_traning=False, layer_num=1, keep_prob=1.0):
    with tf.variable_scope(name):
        # 将输入的数据转成2维进行计算，计算后的结果作为隐藏层输入
        # reshaped = tf.reshape(data, [-1, n_input])
        # 由于使用了relu作为激活函数，所以权重使用He初始值，He初始值使用标准差 √2/n(n为前一层的数量)为的高斯分布
        # 当激活函数为sigmoid或tanh等S型曲线函数时，权重使用Xavier初始值，Xavier初始值使用标准差 √1/n(n为前一层的数量)为的高斯分布
        # weight_in = tf.Variable(tf.random_normal([n_input, n_hidden]))
        # weight_out = tf.Variable(tf.random_normal([n_hidden, n_output]))
        # biase_in = tf.Variable(tf.constant(0.1, shape=[n_hidden, ]))
        # biase_out = tf.Variable(tf.constant(0.1, shape=[n_output, ]))
        # 输入rnn之前先加一层线性变换，可选
        # X_in = tf.nn.dropout(reshaped, keep_prob)
        # X_in = tf.matmul(X_in, weight_in) + biase_in
        # X_in = tf.nn.relu(X_in)
        # 将数据转为3维，作为LSTM_cell的输入
        X_in = tf.reshape(data, [-1, n_steps, n_hidden])
        # 定义一层 LSTM_cell，只需要说明 n_hidden, 它会自动匹配输入的 X 的维度
        lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        # 添加 dropout layer, 一般只设置 output_keep_prob
        lstm_cell = rnn.DropoutWrapper(cell=lstm_cell, input_keep_prob=1.0, output_keep_prob=keep_prob)
        # 调用 MultiRNNCell 来实现多层 LSTM
        lstm_cell = rnn.MultiRNNCell([lstm_cell] * layer_num, state_is_tuple=True)
        # 用全零来初始化state
        init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
        # 调用 dynamic_rnn() 来让我们构建好的网络运行起来
        # 当 time_major==False 时， outputs.shape = [batch_size, n_steps, n_hidden]
        # 所以，可以取 h_state = outputs[:, -1, :] 作为最后输出
        # state.shape = [layer_num, 2, batch_size, n_hidden],
        # 或者，可以取 h_state = state[-1][1] 作为最后输出
        # 最后输出维度是 [batch_size, n_hidden]
        # outputs, finnal_state = tf.nn.dynamic_rnn(lstm_cell, inputs=X_in, initial_state=init_state, time_major=False)
        outputs, finnal_state = tf.nn.dynamic_rnn(lstm_cell, inputs=X_in, dtype=tf.float32, time_major=False)
        # X_out = tf.nn.dropout(finnal_state[-1][1], keep_prob)
        # last_state = tf.matmul(X_out, weight_out) + biase_out
        # last_state = tf.nn.relu(last_state)
        last_state = finnal_state[-1][1]
        if is_traning is not None and regularizer is not None:
            # tf.trainable_variables() 得到所有可以训练的参数
            tf.add_to_collection('losses', tf.reduce_sum([regularizer(v) for v in tf.trainable_variables()]))
        # weight_in_mean = tf.reduce_mean(weight_in)
        # tf.summary.scalar('weight_in_mean', weight_in_mean)
        # tf.summary.histogram('weight_in_mean', weight_in_mean)
        # weight_out_mean = tf.reduce_mean(weight_out)
        # tf.summary.scalar('weight_out_mean', weight_out_mean)
        # tf.summary.histogram('weight_out_mean', weight_out_mean)
    return last_state


# 定义全链接层
def fc_net(name, inputdata, w_shape, regularizer=False, is_traning=False):
    with tf.variable_scope(name):
        weight = tf.get_variable('weights', w_shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
        if is_traning is not None and regularizer is not None:
            tf.add_to_collection('losses', regularizer(weight))
        biases = tf.get_variable('biases', w_shape[-1], initializer=tf.constant_initializer(0.1))
        conve = tf.matmul(inputdata, weight) + biases
        # 生成可视化图
        tf.summary.histogram(name + '_weight', weight)
        mean_name = tf.reduce_mean(weight)
        tf.summary.scalar(name + '_mean', mean_name)
        stddev_name = tf.sqrt(tf.reduce_mean(tf.square(weight - mean_name)))
        tf.summary.scalar(name + '_stddev', stddev_name)
    return conve


# 定义损失函数
def losses(logits, labels, class_num):
    with tf.variable_scope('loss') as scope:
        logits = tf.reshape(logits, [-1, class_num])
        labels = tf.reshape(labels, [-1, class_num])
        # cross_entropy = -tf.reduce_sum(labels * tf.log(tf.clip_by_value(logits, 1e-10, 1.0)))
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels)
        # cross_entropy = tf.pow(logits - labels, 2)
        # cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
        ses_loss = tf.reduce_mean(cross_entropy)
        tf.add_to_collection('losses', ses_loss)
        loss = tf.add_n(tf.get_collection('losses'))
        tf.summary.scalar('loss', loss)
    return loss


# 定义梯度下降函数
def trainning(loss, learning_rate):
    with tf.name_scope('optimizer'):
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        # optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
        # optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
        optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate)
        # optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
        # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss)
    return train_op


# 定义模型的准确率
def evaluation(logits, labels, class_num):
    with tf.variable_scope('accuracy') as scope:
        logits = tf.reshape(logits, [-1, class_num])
        labels = tf.reshape(labels, [-1, class_num])
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        # correct_prediction = tf.equal(tf.round(logits * 33), tf.round(labels * 33))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)
    return accuracy
