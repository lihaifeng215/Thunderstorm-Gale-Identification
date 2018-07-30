import csv
import numpy as np
import tensorflow as tf
import time
from sklearn.preprocessing import MinMaxScaler



def gen_batch(data_13x13):
    all_batch_data = np.array(data_13x13[:, 5:-1])
    print("all_batch_data.shape:", all_batch_data.shape)
    all_batch_data = all_batch_data.reshape(data_13x13.shape[0], 40, 13, 13)
    all_batch_data = all_batch_data.transpose(0, 2, 3, 1)
    all_batch_label = data_13x13[:, -1]
    return np.array(all_batch_data), np.array(all_batch_label)


def gen_batch1(data_7x7):
    all_batch_data = np.array(data_7x7[:, 5:-1])
    print("all_batch_data.shape:", all_batch_data.shape)
    all_batch_data = all_batch_data.reshape(data_7x7.shape[0], 40, 7, 7)
    all_batch_data = all_batch_data.transpose(0, 2, 3, 1)
    all_batch_label = data_7x7[:, -1]
    return np.array(all_batch_data), np.array(all_batch_label)


'''
improved visible
'''
def preprocess(data, size, channel):
    data_pre = np.reshape(data, [-1, channel])
    data_pre = MinMaxScaler().fit_transform(data_pre)
    data_pre = np.reshape(data_pre, [data.shape[0], size * 2 + 1, size * 2 + 1, channel])
    return data_pre


'''
split training data and testing data
'''

#
# def split_train_and_test(batch_data, batch_label, test_number=500):
#     test_start = -test_number
#     train_x = batch_data[:test_start]
#     train_y = batch_label[:test_start]
#     test_x = batch_data[test_start:]
#     test_y = batch_label[test_start:]
#     return train_x, train_y, test_x, test_y


class Model(object):
    def __init__(self, learning_rate=0.0001):
        self.learning_rate = learning_rate
        self.data = tf.placeholder(tf.float32, shape=[None, 13, 13, 40])
        self.data1 = tf.placeholder(tf.float32, shape=[None, 7, 7, 40])
        self.label = tf.placeholder(tf.int64, shape=[None, ])

        self.build_graph()

    def build_graph(self):
        # CNN_weight_bias
        conv1_weight = self.weight_variable([3, 3, 40, 20])
        conv1_bias = self.bias_variable([20, ])
        conv2_weight = self.weight_variable([3, 3, 20, 20])
        conv2_bias = self.bias_variable([20, ])
        conv3_weight = self.weight_variable([3, 3, 20, 20])
        conv3_bias = self.bias_variable([20, ])

        # R2DCNN_weight_bias
        conv1_weight_r = self.weight_variable([3, 3, 60, 40])
        conv1_bias_r = self.bias_variable([40, ])
        conv2_weight_r = self.weight_variable([3, 3, 40, 40])
        conv2_bias_r = self.bias_variable([40, ])
        conv3_weight_r = self.weight_variable([3, 3, 40, 40])
        conv3_bias_r = self.bias_variable([40, ])

        fc_weight = self.weight_variable([40, num_class])
        fc_weight1 = self.weight_variable([20, num_class])
        fc_bias = self.bias_variable([num_class, ])

        l2_reg = tf.constant(0.75, tf.float32, [1, ])

        # 7 * 7 -> 5 * 5 -> 3 * 3 -> 1 * 1
        conv1 = tf.nn.conv2d(self.data1, conv1_weight, [1, 1, 1, 1], padding='VALID') + conv1_bias
        relu1 = tf.nn.relu(conv1)
        conv2 = tf.nn.conv2d(relu1, conv2_weight, [1, 1, 1, 1], padding='VALID') + conv2_bias
        relu2 = tf.nn.relu(conv2)
        conv3 = tf.nn.conv2d(relu2, conv3_weight, [1, 1, 1, 1], padding='VALID') + conv3_bias
        relu3 = tf.nn.relu(conv3)

        # 13 * 13 -> 11 * 11 -> 9 * 9 -> 7 * 7
        conv1_w = tf.nn.conv2d(self.data, conv1_weight, [1, 1, 1, 1], padding='VALID') + conv1_bias
        relu1_w = tf.nn.relu(conv1_w)
        conv2_w = tf.nn.conv2d(relu1_w, conv2_weight, [1, 1, 1, 1], padding='VALID') + conv2_bias
        relu2_w = tf.nn.relu(conv2_w)
        conv3_w = tf.nn.conv2d(relu2_w, conv3_weight, [1, 1, 1, 1], padding='VALID') + conv3_bias
        relu3_w = tf.nn.relu(conv3_w)

        relu4 = tf.concat([relu3_w, self.data1], 3)

        # R2DCNN    tf.concat(13 * 13 ->(7 * 7),  7 * 7) -> 1 * 1
        conv1_r = tf.nn.conv2d(relu4, conv1_weight_r, [1, 1, 1, 1], padding='VALID') + conv1_bias_r
        relu1_r = tf.nn.relu(conv1_r)
        conv2_r = tf.nn.conv2d(relu1_r, conv2_weight_r, [1, 1, 1, 1], padding='VALID') + conv2_bias_r
        relu2_r = tf.nn.relu(conv2_r)
        conv3_r = tf.nn.conv2d(relu2_r, conv3_weight_r, [1, 1, 1, 1], padding='VALID') + conv3_bias_r
        relu3_r = tf.nn.relu(conv3_r)

        # 7 * 7 CNN
        flatten1 = tf.reshape(relu3, [-1, 20])
        dropout1 = tf.nn.dropout(flatten1, keep_prob=0.1)
        self.logits1 = tf.nn.xw_plus_b(dropout1, fc_weight1, fc_bias)
        self.loss1 = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label,
                                                           logits=self.logits1) + l2_reg * tf.nn.l2_loss(fc_weight1))

        # tf.concat(13 * 13 ->(7 * 7),  7 * 7) CNN
        flatten = tf.reshape(relu3_r, [-1, 40])
        dropout = tf.nn.dropout(flatten, keep_prob=0.1)
        self.logits2 = tf.nn.xw_plus_b(dropout, fc_weight, fc_bias)
        self.loss2 = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label,
                                                           logits=self.logits2) + l2_reg * tf.nn.l2_loss(fc_weight))

        # loss = loss1 + loss2
        self.loss = self.loss1 + self.loss2

        # get the index of maximum possibility(logits: rows:None, cols:2)
        # !!!最终预测时只是使用logits2!!!
        self.prediction = tf.argmax(self.logits2, 1)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.prediction, self.label), tf.float32))
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def weight_variable(self, shape):
        return tf.Variable(tf.truncated_normal(shape, stddev=0.1), name='weight')

    def bias_variable(self, shape):
        return tf.Variable(tf.zeros(shape), name='bias')


if __name__ == '__main__':
    train_data_13x13 = np.loadtxt("file/train_test/train_dataset_17y_Radar_denoised_13x13_append_features_cls_labeled.csv",
                            delimiter=',')
    print("train_data_13x13.shape:", train_data_13x13.shape)
    test_data_13x13 = np.loadtxt("file/train_test/test_dataset_17y_Radar_denoised_13x13_append_features_cls_labeled.csv",
                                  delimiter=',')
    train_data_7x7 = np.loadtxt("file/train_test/train_dataset_17y_Radar_denoised_13x13_append_features_cls_labeled_sub_7x7.csv",
                          delimiter=',')
    print("train_data_7x7.shape:", train_data_7x7.shape)
    test_data_7x7 = np.loadtxt("file/train_test/test_dataset_17y_Radar_denoised_13x13_append_features_cls_labeled_sub_7x7.csv",
        delimiter=',')


    ls3_filename = 'ls3.txt'
    rcnn2_filename = 'rcnn2.txt'
    pred3_filename = 'pred3.txt'
    truth3_filename = 'truth3.txt'
    batch_size = 10
    learning_rate = 0.00001
    num_class = 2
    num_epoch = 500
    step = 100
    tf.set_random_seed(123)

    train_batch_data_13x13, train_batch_label_13x13 = gen_batch(train_data_13x13)
    print("train_batch_data_13x13, train_batch_label_13x13:", train_batch_data_13x13.shape, train_batch_label_13x13.shape)
    test_batch_data_13x13, test_batch_label_13x13 = gen_batch(test_data_13x13)
    print("test_batch_data_13x13, test_batch_label_13x13:", test_batch_data_13x13.shape,
          test_batch_label_13x13.shape)


    train_batch_data_7x7, train_batch_label_7x7 = gen_batch1(train_data_7x7)
    print("train_batch_data_7x7, train_batch_label_7x7:", train_batch_data_7x7.shape, train_batch_label_7x7.shape)
    test_batch_data_7x7, test_batch_label_7x7 = gen_batch1(test_data_7x7)
    print("test_batch_data_7x7, test_batch_label_7x7:", test_batch_data_7x7.shape, test_batch_label_7x7.shape)

    train_batch_data = preprocess(train_batch_data_13x13, size=6,channel=40)
    test_batch_data = preprocess(test_batch_data_13x13, size=6, channel=40)
    train_batch_data1 = preprocess(train_batch_data_7x7, size=3, channel=40)
    test_batch_data1 = preprocess(test_batch_data_7x7, size=3, channel=40)

    val_batch_data, val_batch_label = test_batch_data, test_batch_label_13x13
    val_batch_data1, val_batch_label = test_batch_data1, test_batch_label_7x7

    print(train_batch_data.shape, test_batch_label_13x13.shape, test_batch_data.shape, test_batch_label_13x13.shape)
    print(train_batch_data1.shape, test_batch_label_7x7.shape, test_batch_data1.shape, test_batch_label_7x7.shape)

    rand_ix = np.random.permutation(len(train_batch_data))
    train_batch_data, train_batch_data1, train_batch_label = train_batch_data[rand_ix], train_batch_data1[rand_ix], \
                                                             train_batch_label_13x13[rand_ix]
    rand_ix1 = np.random.permutation(len(val_batch_data))
    val_batch_data, val_batch_data1, val_batch_label = val_batch_data[rand_ix1], val_batch_data1[rand_ix1], \
                                                       val_batch_label[rand_ix1]

    model = Model(learning_rate)
    init = tf.initialize_all_variables()
    saver = tf.train.Saver()
    rcnn2 = []
    ls3 = []
    with tf.Session() as sess:
        sess.run(init)

        for i_epoch in range(num_epoch):

            # training step
            total_train_loss = 0.
            total_train_acc = 0.
            for i in range(0, len(train_batch_data), batch_size):
                if i + batch_size >= len(train_batch_data):
                    break
                batch_data = train_batch_data[i:i + batch_size]

                batch_data1 = train_batch_data1[i:i + batch_size]


                batch_label = train_batch_label[i:i + batch_size]

                _, loss, accuracy = sess.run([model.train_op, model.loss, model.accuracy],
                                             feed_dict={model.data: batch_data, model.data1: batch_data1,
                                                        model.label: batch_label})
                total_train_loss += loss
                total_train_acc += accuracy
            cnt = len(train_batch_data) / batch_size
            print('Epoch rcnn2d', i_epoch, 'train: ', total_train_loss / cnt, total_train_acc / cnt)

            # validation step
            total_val_loss = 0.
            total_val_acc = 0.
            for i in range(0, len(val_batch_data), batch_size):
                if i + batch_size >= len(val_batch_data):
                    break
                batch_data = val_batch_data[i:i + batch_size]
                # print(np.array(batch_data).shape)
                batch_data1 = val_batch_data1[i:i + batch_size]
                batch_label = val_batch_label[i:i + batch_size]

                loss, accuracy = sess.run([model.loss, model.accuracy],
                                          feed_dict={model.data: batch_data, model.data1: batch_data1,
                                                     model.label: batch_label})
                total_val_loss += loss
                total_val_acc += accuracy

            cnt = len(val_batch_data) / batch_size
            print('Epoch rcnn2d', i_epoch, 'val: ', total_val_loss / cnt, total_val_acc / cnt)
            ls3.append(total_train_loss / step)
            np.savetxt(ls3_filename, ls3, delimiter=',', fmt='%f')

            if (total_val_acc / cnt) >= 0.85:
                save_path = saver.save(sess,"file/save_net.ckpt")
                print("save to path:",save_path)
                break

        print("~~~~~~~~~~~~~~~~~Optimization Finished!~~~~~~~~~~~~~~~~~~~~~")
        print("Testing Accuracy:", )
        pred3 = []
        truth3 = []
        all_batch_label = []
        start = time.time()

        for i in range(0, len(val_batch_data), batch_size):
            if i + batch_size >= len(val_batch_data):
                break
            batch_data = test_batch_data[i:i + batch_size]
            batch_data1 = test_batch_data1[i:i + batch_size]
            batch_label = test_batch_label_13x13[i:i + batch_size]
            prediction = sess.run([model.prediction],
                                  feed_dict={model.data: batch_data, model.data1: batch_data1,
                                             model.label: batch_label})
            prediction = np.reshape(prediction, [batch_size, 1])
            all_batch_label = np.reshape(batch_label, [batch_size, 1])
            pred3.append(prediction)
            truth3.append(all_batch_label)
        print("use time:", time.time()-start)
        np.savetxt("pred3.csv", np.reshape(pred3, (-1,)), delimiter=',', fmt='%d')
        np.savetxt("truth3.csv", np.reshape(truth3, (-1,)), delimiter=',', fmt='%d')

