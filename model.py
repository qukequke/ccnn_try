import tensorflow as tf
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import os
import scipy.misc
import glob


def get_weight(shape):
    return tf.Variable(tf.truncated_normal(shape=shape, stddev=0.1), dtype=tf.float32)


def get_bias(shape):
    return tf.Variable(tf.zeros(shape=shape, dtype=tf.float32))


def block(inputs):
    w1 = get_weight([3, 3, 1, 32])
    b1 = get_bias([32])
    w2 = get_weight([3, 3, 32, 64])
    b2 = get_bias([64])
    w3 = get_weight([3, 3, 64, 24])
    b3 = get_bias([24])
    w4 = get_weight([3, 3, 24, 1])
    b4 = get_bias([1])
    conv1 = tf.nn.conv2d(input=inputs, filter=w1, strides=[1, 1, 1, 1], padding='SAME') + b1
    s_conv1 = tf.nn.relu(conv1)
    conv2 = tf.nn.conv2d(input=s_conv1, filter=w2, strides=[1, 1, 1, 1], padding='SAME') + b2
    s_conv2 = tf.nn.relu(conv2)
    conv3 = tf.nn.conv2d(input=s_conv2, filter=w3, strides=[1, 1, 1, 1], padding='SAME') + b3
    s_conv3 = tf.nn.relu(conv3)
    conv4 = tf.nn.conv2d(input=s_conv3, filter=w4, strides=[1, 1, 1, 1], padding='SAME')
    # s_conv4 = tf.nn.sigmoid(conv4)
    return conv4


def abs_layer(real, imag):
    return tf.sqrt(tf.square(real) + tf.square(imag))


def get_loss(outputs, label):
    # loss = tf.reduce_mean(tf.square(tf.abs(outputs-label)))
    loss = 100.0 * tf.reduce_mean(tf.square(tf.abs(outputs-label)))
    return loss


def save(sess, path):
    dir_name = os.path.dirname(path)
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    saver = tf.train.Saver()
    saver.save(sess, path)


def load_data(path):
    print(path)
    print(glob.glob(path))
    filename = glob.glob(path)[0]
    data_label = sio.loadmat(filename)
    datas = data_label['data']
    labels = data_label['label']
    return datas, labels


def util_data_complex(datas, labels):
    global idx
    idx += 1
    if idx >= datas.shape[-1]:
        idx %= datas.shape[-1]
    # idx = next_data(idx)
    data = datas[:, :, :, idx]
    label = labels[:, :, idx]
    data = normalization(data)
    label = normalization(label)
    data_real = data[:, :, 0]
    data_imag = data[:, :, 1]
    return np.expand_dims(data_real, 0), np.expand_dims(data_imag, 0), np.expand_dims(label, 0)


def get_batch_data(all_data, all_label, batch_size):
    datas = np.zeros(shape=(batch_size, 140, 140, 2))
    labels = np.zeros(shape=(batch_size, 140, 140, 1))
    for i in range(batch_size):
        datas[i, :, :, 0], datas[i, :, :, 1], labels[i, :, :, 0] = util_data_complex(all_data, all_label)
    return datas, labels


def util_data(datas, labels, idx):
    if idx >= datas.shape[-1]:
        idx %= datas.shape[-1]
    # idx = next_data(idx)
    data = datas[:, :, :, idx]
    label = labels[:, :, idx]
    # data = normalization(data)
    # label = normalization(label)
    return np.expand_dims(data, 0), np.expand_dims(np.expand_dims(label, 0), 3)


def normalization(data):
    min_ = data.min()
    max_ = data.max()
    data = (data - min_) / (max_ - min_)
    return data


def save_prediction(sess, outputs, x_real, x_imag, test_y, step, prediction_path):
    data = np.sqrt(x_real[0, :, :, 0] ** 2 + x_imag[0, :, :, 0] ** 2)

    prediction = sess.run(outputs, feed_dict={inputs1: x_real, inputs2: x_imag})
    # save_img = np.concatenate([data, prediction[0, :, :, 0], test_y[0, :, :, 0]], axis=1)
    # plt.imshow(save_img)
    # plt.show()
    # print('predicion > 0.1 number is ')
    # print((prediction>0.1).sum())
    plt.imsave(os.path.join(prediction_path ,str(step)) + '.png', prediction[0, :, :, 0])
    plt.imsave(os.path.join(prediction_path ,str(step)) + 'y.png', test_y[0, :, :, 0])
    plt.imsave(os.path.join(prediction_path ,str(step)) + 'x.png', np.sqrt(x_real[0, :, :, 0] ** 2 + x_imag[0, :, :, 0]))


inputs1 = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 1])
inputs2 = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 1])
labels = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 1])
output_real = block(inputs1)
output_imag = block(inputs2)
outputs = abs_layer(output_real, output_imag)
# print(len(tf.trainable_variables()))
# var1 = tf.trainable_variables()[:8]
# var2 = tf.trainable_variables()[8:]
# lr1 = 3 * (10 ** (-3))
lr = 10 ** (-3)
# lr1 = tf.Variable(lr1, dtype=tf.float32)
# lr2 = tf.Variable(lr2, dtype=tf.float32)
cost = get_loss(outputs, labels)
sum_loss = tf.summary.scalar('loss', cost)
# train_op1 = tf.train.MomentumOptimizer(lr1, 0.9).minimize(cost, var_list=var1)
# train_op2 = tf.train.MomentumOptimizer(lr2, 0.9).minimize(cost, var_list=var2)
# train_op = tf.group(train_op1, train_op2)
train_op = tf.train.AdamOptimizer(lr, 0.9).minimize(cost)

output_path = 'log/'
save_path = os.path.join(output_path, 'model.ckpt')
epochs = 50
training_iters = 1000
prediction_path = './prediction'
restore = False
tstep = tf.Variable(0, trainable=False, name='tstep')
train_path = 'train_data/*.mat'
test_path = 'test_data/*.mat'
idx = -2

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # writer = tf.summary.FileWriter('graph/', sess.graph)
    if restore == True:
        ckpt = tf.train.get_checkpoint_state(output_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver = tf.train.Saver()
            print(ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
    cur_step = tstep.eval()
    cur_epoch = cur_step // training_iters
    print('cur_step = ' + str(cur_step))
    print('cur_epoch = ' + str(cur_epoch))


    # test_x_real, test_x_imag, test_y = util_data_complex(all_datas, all_labels)

    all_test_datas, all_test_labels = load_data(test_path)

    batch_data, test_y = get_batch_data(all_test_datas, all_test_labels, 1)
    test_x_real = np.expand_dims(batch_data[:, :, :, 0], axis=3)
    test_x_imag = np.expand_dims(batch_data[:, :, :, 0], axis=3)

    all_datas, all_labels = load_data(train_path)
    # real_test_x_real, real_test_x_imag, real_test_y = util_data_complex(all_test_datas, all_test_labels)
    # test_x, test_y = util_data(all_datas, all_labels, 0)

    for epoch in range(epochs):
        total_loss = 0
        # sess.run(lr1.assign(lr1.eval() * (1 ** epoch)))
        # sess.run(lr2.assign(lr2.eval() * (1 ** epoch)))
        # print('lr1 = ' + str(lr1.eval()))
        # print('lr2 = ' + str(lr2.eval()))
        for step in range(np.maximum(epoch*training_iters, cur_step), ((epoch+1)*training_iters)):
            # data_real, data_imag, label = util_data_complex(all_datas, all_labels, step)
            batch_data, label = get_batch_data(all_datas, all_labels, 4)
            data_real = np.expand_dims(batch_data[:, :, :, 0], axis=3)
            data_imag = np.expand_dims(batch_data[:, :, :, 1], axis=3)
            # data_con = data[0, :, :, 0]
            # label_con = label[0, :, :, 0]
            # print(label.shape)
            # fig, ax = plt.subplots(1, 2)
            # ax[0].imshow(np.sqrt(data[0, :, :, 0] ** 2 + data[0, :, :, 1] ** 2))
            # ax[1].imshow(label[0, :, :, 0])
            # plt.show()

            _, loss, prediction, summary = sess.run([train_op, cost, outputs, sum_loss], feed_dict={inputs1: data_real, inputs2: data_imag, labels: label})
            # print('prediction min is ' + str(prediction.min()))
            # print('prediction max is ' + str(prediction.max()))
            # writer.add_summary(summary, step)

            predi_cont = prediction[0, :, :, 0]
            tstep.assign(step).eval()
            total_loss += loss
            if step % 20 == 0:
                print('loss = ' + str(loss) + '   step = ' + str(step) + '   epoch = ' + str(epoch))

        print('average loss = ' + str(total_loss / training_iters))
        save(sess, save_path)
        save_prediction(sess, outputs, test_x_real, test_x_imag, test_y, epoch, prediction_path)
        # save_prediction(sess, outputs, test_x_real, test_x_imag, test_y, epoch, prediction_path)
        # save_prediction(sess, outputs, real_test_x_real, real_test_x_imag, real_test_y, epoch, 'prediction1')

    print('optimizer finished')

