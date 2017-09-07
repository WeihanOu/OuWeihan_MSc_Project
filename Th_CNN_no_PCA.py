import tensorflow as tf
import numpy as np
from ReadImg import *
from ExtractFeature import *
import random
import tensorflow
#
# learn_rate = 0.001
# batch_size = 60
#
# n_hidden_units = 64
#
eigen_height = 130
eigen_width = 100
#
# n_relax = 740
# n_stress_1 = 1573
# n_stress_2 = 1195
#
x_in = tf.placeholder(tf.float32, [None, eigen_height, eigen_width])
y_in = tf.placeholder(tf.float32, [None, 3])

def conv2d(x_in, w):
    return tf.nn.conv2d(x_in, w, strides=[1, 1, 1, 1], padding='SAME')


def conv2d_not_same(x_in, w):
    return tf.nn.conv2d(x_in, w, strides=[1, 1, 1, 1], padding='VALID')


def max_pool_2x2(x_in):
    return tf.nn.max_pool(x_in, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def max_pool_1x2(x_in):
    return tf.nn.max_pool(x_in, ksize=[1, 1, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
#
#
# def get_model(x):
#     # 720 1280 -> 360 640
#     W_conv01 = tf.Variable(tf.truncated_normal([3, 3, 3, 16], stddev=0.1))
#     b_conv01 = tf.Variable(tf.constant(0.1, shape=[16]))
#     W_conv02 = tf.Variable(tf.truncated_normal([3, 3, 3, 16], stddev=0.1))
#     b_conv02 = tf.Variable(tf.constant(0.1, shape=[16]))
#
#     W_conv03 = tf.Variable(tf.truncated_normal([3, 3, 3, 32], stddev=0.1))
#     b_conv03 = tf.Variable(tf.constant(0.1, shape=[32]))
#     W_conv04 = tf.Variable(tf.truncated_normal([3, 3, 3, 32], stddev=0.1))
#     b_conv04 = tf.Variable(tf.constant(0.1, shape=[32]))
#
#     W_conv1 = tf.Variable(tf.truncated_normal([3, 3, 3, 64], stddev=0.1))
#     b_conv1 = tf.Variable(tf.constant(0.1, shape=[64]))
#     # 360 640 -> 180 320
#     # W_conv2 = tf.Variable(tf.truncated_normal([3, 3, 64, 128], stddev=0.1))
#     # b_conv2 = tf.Variable(tf.constant(0.1, shape=[32]))
#
#     W_conv2 = tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev=0.1))
#     b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))
#     # 180 320 -> 90 160======================
#     W_conv3 = tf.Variable(tf.truncated_normal([3, 3, 64, 128], stddev=0.1))
#     b_conv3 = tf.Variable(tf.constant(0.1, shape=[128]))
#     W_conv4 = tf.Variable(tf.truncated_normal([3, 3, 128, 128], stddev=0.1))
#     b_conv4 = tf.Variable(tf.constant(0.1, shape=[128]))
#     # 90 160 -> 45 80
#     # W_conv4 = tf.Variable(tf.truncated_normal([3, 3, 32, 64], stddev=0.1))
#     # b_conv4 = tf.Variable(tf.constant(0.1, shape=[64]))
#     W_conv5 = tf.Variable(tf.truncated_normal([3, 3, 128, 256], stddev=0.1))
#     b_conv5 = tf.Variable(tf.constant(0.1, shape=[256]))
#     W_conv6 = tf.Variable(tf.truncated_normal([3, 3, 256, 256], stddev=0.1))
#     b_conv6 = tf.Variable(tf.constant(0.1, shape=[256]))
#     W_conv7 = tf.Variable(tf.truncated_normal([3, 3, 256, 256], stddev=0.1))
#     b_conv7 = tf.Variable(tf.constant(0.1, shape=[256]))
#
#     W_conv8 = tf.Variable(tf.truncated_normal([3, 3, 256, 512], stddev=0.1))
#     b_conv8 = tf.Variable(tf.constant(0.1, shape=[512]))
#     W_conv9 = tf.Variable(tf.truncated_normal([3, 3, 512, 512], stddev=0.1))
#     b_conv9 = tf.Variable(tf.constant(0.1, shape=[512]))
#     W_conv10 = tf.Variable(tf.truncated_normal([3, 3, 512, 512], stddev=0.1))
#     b_conv10 = tf.Variable(tf.constant(0.1, shape=[512]))
#     # 45 80 -> 44 80
#     # W_conv5 = tf.Variable(tf.truncated_normal([2, 1, 64, 64], stddev=0.1))
#     # b_conv5 = tf.Variable(tf.constant(0.1, shape=[64]))
#     W_conv11 = tf.Variable(tf.truncated_normal([3, 3, 512, 512], stddev=0.1))
#     b_conv11 = tf.Variable(tf.constant(0.1, shape=[512]))
#     W_conv12 = tf.Variable(tf.truncated_normal([3, 3, 512, 512], stddev=0.1))
#     b_conv12 = tf.Variable(tf.constant(0.1, shape=[512]))
#     W_conv13 = tf.Variable(tf.truncated_normal([3, 3, 512, 512], stddev=0.1))
#     b_conv13 = tf.Variable(tf.constant(0.1, shape=[512]))
#
#     x_image = tf.reshape(x, [-1, eigen_height, eigen_width, 1])
#
#     # 130 100 -> 65 50
#     c01 = tf.nn.relu(conv2d(x_image, W_conv01) + b_conv01)
#     c02 = tf.nn.relu(conv2d(c01, W_conv02) + b_conv02)
#     p02 = max_pool_2x2(c02)
#     # 65 50 -> 33 25
#     c03 = tf.nn.relu(conv2d(p02, W_conv03) + b_conv03)
#     c04 = tf.nn.relu(conv2d(c03, W_conv04) + b_conv04)
#     p04 = max_pool_2x2(c04)
#     # 33 25 -> 17 13
#     c1 = tf.nn.relu(conv2d(p04, W_conv1) + b_conv1)
#     c2 = tf.nn.relu(conv2d(c1, W_conv2) + b_conv2)
#     p2 = max_pool_2x2(c2)
#     # 17 13 -> 9 7
#     c3 = tf.nn.relu(conv2d(p2, W_conv3) + b_conv3)
#     c4 = tf.nn.relu(conv2d(c3, W_conv4) + b_conv4)
#     p4 = max_pool_2x2(c4)
#     # 9 7 -> 5 4
#     c5 = tf.nn.relu(conv2d(p4, W_conv5) + b_conv5)
#     c6 = tf.nn.relu(conv2d_not_same(c5, W_conv6) + b_conv6)
#     c7 = tf.nn.relu(conv2d(c6, W_conv7) + b_conv7)
#     p7 = max_pool_2x2(c7)
#     # 5 4 -> 3 2
#     c8 = tf.nn.relu(conv2d(p7, W_conv8) + b_conv8)
#     c9 = tf.nn.relu(conv2d(c8, W_conv9) + b_conv9)
#     c10 = tf.nn.relu(conv2d(c9, W_conv10) + b_conv9)
#     p10 = max_pool_2x2(c10)
#     # 3 2 -> 2 1
#     c11 = tf.nn.relu(conv2d(p10, W_conv11) + b_conv11)
#     c12 = tf.nn.relu(conv2d(c11, W_conv12) + b_conv12)
#     c13 = tf.nn.relu(conv2d(c12, W_conv13) + b_conv13)
#     p13 = max_pool_2x2(c13)
#     # 2 1 -> 1 1
#     c130 = tf.nn.relu(conv2d(p13, W_conv11) + b_conv11)
#     c131 = tf.nn.relu(conv2d(c130, W_conv12) + b_conv12)
#     c132 = tf.nn.relu(conv2d(c131, W_conv13) + b_conv13)
#     p132 = max_pool_2x2(c132)
#
#     p13_flat = tf.reshape(p132, [-1, 512])
#
#     W_flat_1 = tf.Variable(tf.truncated_normal([512, 1024], stddev=0.1))
#     b_flat_1 = tf.Variable(tf.constant(0.1, shape=[1024]))
#     W_flat_2 = tf.Variable(tf.truncated_normal([1024, 256], stddev=0.1))
#     b_flat_2 = tf.Variable(tf.constant(0.1, shape=[256]))
#     W_flat_out = tf.Variable(tf.truncated_normal([256, 3], stddev=0.1))
#     b_flat_out = tf.Variable(tf.constant(0.1, shape=[3]))
#
#     h_flat_1 = tf.nn.relu(tf.matmul(p13_flat, W_flat_1) + b_flat_1)
#     h_flat_2 = tf.nn.relu(tf.matmul(h_flat_1, W_flat_2) + b_flat_2)
#     h_flat_out = tf.matmul(h_flat_2, W_flat_out) + b_flat_out
#
#     pro = 0.8
#
#     out = tf.nn.dropout(h_flat_out, pro)
#
#     return out
#
#
# def main():
#     # read img & prepossessing
#     x_tr_0, x_tr_1, x_tr_2 = exam_read_file(file_0, file_1, file_2)
#     for rand_i in range(10):
#         print("i: ", rand_i)
#         idx_1, idx_2 = choose_img()
#         x_tr_1 = np.reshape(x_tr_1[:, :, idx_1], [130, 100, 3695])
#         x_tr_2 = np.reshape(x_tr_2[:, :, idx_2], [130, 100, 3695])
#         # face_all = np.concatenate((x_tr_0, x_tr_1,x_tr_2))
#         # face_all = np.transpose(face_all, [2, 0, 1])
#
#         y_tr = np.zeros(int(np.size(x_tr_0, 2) + np.size(x_tr_1, 2) + np.size(x_tr_2, 2)) - 740 * 3)
#         y_tr[0:int(np.size(x_tr_0, 2) - 740)] = 0
#         y_tr[int(np.size(x_tr_0, 2) - 740):int(np.size(x_tr_0, 2) - 740) + int(np.size(x_tr_1, 2) - 740)] = 1
#         y_tr[int(np.size(x_tr_0, 2) - 740) + int(np.size(x_tr_1, 2) - 740):
#         int(np.size(x_tr_0, 2) - 740) + int(np.size(x_tr_1, 2) - 740) + int(np.size(x_tr_2, 2) - 740)] = 2
#
#         for fold_j in range(5):
#             x_te_0 = np.concatenate((x_tr_0[:, :, 0 * 740 + 148 * fold_j:0 * 740 + 148 * fold_j + 148],
#                                      x_tr_0[:, :, 1 * 740 + 148 * fold_j:1 * 740 + 148 * fold_j + 148],
#                                      x_tr_0[:, :, 2 * 740 + 148 * fold_j:2 * 740 + 148 * fold_j + 148],
#                                      x_tr_0[:, :, 3 * 740 + 148 * fold_j:3 * 740 + 148 * fold_j + 148],
#                                      x_tr_0[:, :, 4 * 740 + 148 * fold_j:4 * 740 + 148 * fold_j + 148]), axis=2)
#
#             x_te_1 = np.concatenate((x_tr_1[:, :, 0 * 740 + 148 * fold_j:0 * 740 + 148 * fold_j + 148],
#                                      x_tr_1[:, :, 1 * 740 + 148 * fold_j:1 * 740 + 148 * fold_j + 148],
#                                      x_tr_1[:, :, 2 * 740 + 148 * fold_j:2 * 740 + 148 * fold_j + 148],
#                                      x_tr_1[:, :, 3 * 740 + 148 * fold_j:3 * 740 + 148 * fold_j + 148],
#                                      x_tr_1[:, :, 4 * 740 + 148 * fold_j:4 * 740 + 148 * fold_j + 148]), axis=2)
#
#             x_te_2 = np.concatenate((x_tr_2[:, :, 0 * 740 + 148 * fold_j:0 * 740 + 148 * fold_j + 148],
#                                      x_tr_2[:, :, 1 * 740 + 148 * fold_j:1 * 740 + 148 * fold_j + 148],
#                                      x_tr_2[:, :, 2 * 740 + 148 * fold_j:2 * 740 + 148 * fold_j + 148],
#                                      x_tr_2[:, :, 3 * 740 + 148 * fold_j:3 * 740 + 148 * fold_j + 148],
#                                      x_tr_2[:, :, 4 * 740 + 148 * fold_j:4 * 740 + 148 * fold_j + 148]), axis=2)
#             idx_0 = list(
#                 set(np.arange(np.size(x_tr_0, 2))) - set(np.arange(0 * 740 + 148 * fold_j, 0 * 740 + 148 * fold_j + 148)) -
#                 set(np.arange(1 * 740 + 148 * fold_j, 1 * 740 + 148 * fold_j + 148)) -
#                 set(np.arange(2 * 740 + 148 * fold_j, 2 * 740 + 148 * fold_j + 148)) -
#                 set(np.arange(3 * 740 + 148 * fold_j, 3 * 740 + 148 * fold_j + 148)) -
#                 set(np.arange(4 * 740 + 148 * fold_j, 4 * 740 + 148 * fold_j + 148)))
#
#             idx_1 = list(
#                 set(np.arange(np.size(x_tr_1, 2))) - set(np.arange(0 * 740 + 148 * fold_j, 0 * 740 + 148 * fold_j + 148)) -
#                 set(np.arange(1 * 740 + 148 * fold_j, 1 * 740 + 148 * fold_j + 148)) -
#                 set(np.arange(2 * 740 + 148 * fold_j, 2 * 740 + 148 * fold_j + 148)) -
#                 set(np.arange(3 * 740 + 148 * fold_j, 3 * 740 + 148 * fold_j + 148)) -
#                 set(np.arange(4 * 740 + 148 * fold_j, 4 * 740 + 148 * fold_j + 148)))
#
#             idx_2 = list(
#                 set(np.arange(np.size(x_tr_2, 2))) - set(np.arange(0 * 740 + 148 * fold_j, 0 * 740 + 148 * fold_j + 148)) -
#                 set(np.arange(1 * 740 + 148 * fold_j, 1 * 740 + 148 * fold_j + 148)) -
#                 set(np.arange(2 * 740 + 148 * fold_j, 2 * 740 + 148 * fold_j + 148)) -
#                 set(np.arange(3 * 740 + 148 * fold_j, 3 * 740 + 148 * fold_j + 148)) -
#                 set(np.arange(4 * 740 + 148 * fold_j, 4 * 740 + 148 * fold_j + 148)))
#             x_tr = np.concatenate((x_tr_0[:, :, idx_0], x_tr_1[:, :, idx_1], x_tr_2[:, :, idx_2]), axis=2)
#
#             x_te = np.concatenate((x_te_0, x_te_1, x_te_2), axis=2)
#
#             y_te = np.zeros(2220)
#             y_te[0:740] = 0
#             y_te[740:1480] = 1
#             y_te[1480:2220] = 2
#     # x_tr, x_te, y_tr, y_te
#             x_tr = np.transpose(x_tr, [2, 0, 1])
#             x_te = np.transpose(x_te, [2, 0, 1])
#         model = get_model(x_in)
#
#         loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=y_in), 0)
#         # training with Adam
#         train_step = tf.train.AdamOptimizer().minimize(loss)
#
#         correct_pred = tf.equal(tf.argmax(model, 1), tf.argmax(y_in, 1))
#         accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), 0)
#         # saver = tf.train.Saver()
#         with tf.device('/gpu:0'):
#             with tf.Session() as sess:
#                 sess.run(tf.global_variables_initializer())
#                 for i in range(0, 10000):
#                     # train the model
#
#                     index_0 = random.sample(range(int(n_relax)), 20)
#                     index_1 = random.sample(range(int(n_relax), int(n_relax) + n_stress_1), 20)
#                     index_2 = random.sample(range(int(n_relax) + n_stress_1, int(n_relax) + n_stress_1 + n_stress_2), 20)
#                     # this_x_train = x_train[this_batch_num:this_batch_num + batch_size, :]
#                     # this_y_train = y_train[this_batch_num:this_batch_num + batch_size, :]
#
#                     this_x_train0 = face_train[index_0, :, :]
#                     this_y_train0 = y_train[index_0, :]
#
#                     this_x_train1 = face_train[index_1, :, :]
#                     this_y_train1 = y_train[index_1, :]
#
#                     this_x_train2 = face_train[index_2, :, :]
#                     this_y_train2 = y_train[index_2, :]
#
#                     this_x_train = np.zeros((batch_size, h, w))
#                     this_x_train[0:20, :, :] = this_x_train0
#                     this_x_train[20:40, :, :] = this_x_train1
#                     this_x_train[40:60, :, :] = this_x_train2
#
#                     this_y_train = np.zeros((batch_size, 3))
#                     this_y_train[0:20, :] = this_y_train0
#                     this_y_train[20:40, :] = this_y_train1
#                     this_y_train[40:60, :] = this_y_train2
#
#                     # _, model = sess.run([train_step,model], feed_dict={x_in: this_x_train, y_in: this_y_train})
#                     _, lo, mo = sess.run([train_step, loss, model], feed_dict={x_in: this_x_train, y_in: this_y_train})
#                     if i % 20 == 0:
#                         this_train_loss = sess.run(loss, feed_dict={x_in: this_x_train, y_in: this_y_train})
#                         this_test_loss = sess.run(loss, feed_dict={x_in: face_test, y_in: y_test})
#
#                         this_train_accuracy = sess.run(accuracy, feed_dict={x_in: this_x_train, y_in: this_y_train})
#                         this_test_accuracy = sess.run(accuracy, feed_dict={x_in: face_test, y_in: y_test})
#                         # print('The', i, 'time train accuracy: ', this_train_accuracy)
#                         print('The', i, 'time train loss: ', this_train_loss, ', train accuracy: ', this_train_accuracy)
#                         print('The', i, 'time test loss: ', this_test_loss, ', test accuracy: ', this_test_accuracy)
#
#
# if __name__ == '__main__':
#     main()
#
import glob
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from Preprocess import *
import time
from sklearn.metrics import confusion_matrix

import random
from ExtractFeature import *
from sklearn.svm import LinearSVC

file_0 = './data/new_all_same_size_face/0/*.npy'
file_1 = './data/new_all_same_size_face/1/*.npy'
file_2 = './data/new_all_same_size_face/2/*.npy'
n_img = 785

def exam_read_file(f_0, f_1, f_2):
    # read train data from outside file
    list_0 = glob.glob(file_0)
    # imgs_0 = np.array([np.array(Image.open(fname).convert('L')) for fname in list_0])
    imgs_0 = np.array([np.array(np.load(fname)) for fname in list_0])
    imgs_0 = np.transpose(imgs_0, [1, 2, 0])
    list_1 = glob.glob(file_1)
    imgs_1 = np.array([np.array(np.load(fname)) for fname in list_1])
    imgs_1 = np.transpose(imgs_1, [1, 2, 0])
    list_2 = glob.glob(file_2)
    imgs_2 = np.array([np.array(np.load(fname)) for fname in list_2])
    imgs_2 = np.transpose(imgs_2, [1, 2, 0])
    return imgs_0, imgs_1, imgs_2


def get_model(x_train):

    # 130 100  65 50
    W_conv1 = tf.Variable(tf.truncated_normal([3, 3, 1, 32], stddev=0.1))
    b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]))

    # 65 50  64 50
    W_conv2 = tf.Variable(tf.truncated_normal([2, 1, 32, 64], stddev=0.1))
    b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))

    # 64 50  32 25
    W_conv3 = tf.Variable(tf.truncated_normal([3, 3, 64, 128], stddev=0.1))
    b_conv3 = tf.Variable(tf.constant(0.1, shape=[128]))

    # 32 25  32 24
    W_conv4 = tf.Variable(tf.truncated_normal([1, 2, 128, 256], stddev=0.1))
    b_conv4 = tf.Variable(tf.constant(0.1, shape=[256]))

    # 32 24  16 12
    W_conv5 = tf.Variable(tf.truncated_normal([3, 3, 256, 256], stddev=0.1))
    b_conv5 = tf.Variable(tf.constant(0.1, shape=[256]))

    # 16 12  8 6
    W_conv6 = tf.Variable(tf.truncated_normal([3, 3, 256, 512], stddev=0.1))
    b_conv6 = tf.Variable(tf.constant(0.1, shape=[512]))

    # 8 6  4 3
    W_conv7 = tf.Variable(tf.truncated_normal([3, 3, 512, 512], stddev=0.1))
    b_conv7 = tf.Variable(tf.constant(0.1, shape=[512]))

    # 4 3  3 3
    W_conv8 = tf.Variable(tf.truncated_normal([2, 1, 512, 1024], stddev=0.1))
    b_conv8 = tf.Variable(tf.constant(0.1, shape=[1024]))

    # 3 3  1 1
    W_conv9 = tf.Variable(tf.truncated_normal([3, 3, 1024, 1024], stddev=0.1))
    b_conv9 = tf.Variable(tf.constant(0.1, shape=[1024]))

    x_image = tf.reshape(x_train, [-1, 130, 100, 1])

    # 130 100 -> 65 50
    c1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    p1 = max_pool_2x2(c1)
    # 65 50 -> 64 50
    c2 = tf.nn.relu(conv2d_not_same(p1, W_conv2) + b_conv2)
    # 64 50 -> 32 25
    c3 = tf.nn.relu(conv2d(c2, W_conv3) + b_conv3)
    p3 = max_pool_2x2(c3)
    # 32 25 -> 32 24
    c4 = tf.nn.relu(conv2d_not_same(p3, W_conv4) + b_conv4)
    # 32 24 -> 16 12
    c5 = tf.nn.relu(conv2d(c4, W_conv5) + b_conv5)
    p5 = max_pool_2x2(c5)
    # 16 12 -> 8 6
    c6 = tf.nn.relu(conv2d(p5, W_conv6) + b_conv6)
    p6 = max_pool_2x2(c6)
    # 8 6 -> 4 3
    c7 = tf.nn.relu(conv2d(p6, W_conv7) + b_conv7)
    p7 = max_pool_2x2(c7)
    # 4 3 -> 3 3
    c8 = tf.nn.relu(conv2d_not_same(p7, W_conv8) + b_conv8)
    # 3 3 -> 1 1
    c9 = tf.nn.relu(conv2d_not_same(c8, W_conv9) + b_conv9)

    c9_flat = tf.reshape(c9, [-1, 1024])

    W_flat_1 = tf.Variable(tf.truncated_normal([1024, 1024], stddev=0.1))
    b_flat_1 = tf.Variable(tf.constant(0.1, shape=[1024]))
    # W_flat_2 = tf.Variable(tf.truncated_normal([1024, 256], stddev=0.1))
    # b_flat_2 = tf.Variable(tf.constant(0.1, shape=[256]))
    W_flat_out = tf.Variable(tf.truncated_normal([1024, 3], stddev=0.1))
    b_flat_out = tf.Variable(tf.constant(0.1, shape=[3]))

    h_flat_1 = tf.nn.relu(tf.matmul(c9_flat, W_flat_1) + b_flat_1)
    # h_flat_2 = tf.nn.relu(tf.matmul(h_flat_1, W_flat_2) + b_flat_2)
    out = tf.matmul(h_flat_1, W_flat_out) + b_flat_out

    pro = 0.8

    out = tf.nn.dropout(out, pro)

    return out

def choose_img(x):
    # since stress==0 has smallest number of images, we choose all images with stress==0
    idx_0_0 = np.reshape(random.sample(range(0, 790), n_img), [-1, 1])
    idx_0_1 = np.reshape(random.sample(range(790, 2466), n_img), [-1, 1])
    idx_0_2 = np.reshape(random.sample(range(2466, np.size(x[0], 2)), n_img), [-1, 1])

    idx_1_0 = np.reshape(random.sample(range(0, 833), n_img), [-1, 1])
    idx_1_1 = np.reshape(random.sample(range(833, 2315), n_img), [-1, 1])
    idx_1_2 = np.reshape(random.sample(range(2415, np.size(x[1], 2)), n_img), [-1, 1])

    idx_2_0 = np.reshape(random.sample(range(0, 790), n_img), [-1, 1])
    idx_2_1 = np.reshape(random.sample(range(790, 2466), n_img), [-1, 1])
    idx_2_2 = np.reshape(random.sample(range(2466, np.size(x[2], 2)), n_img), [-1, 1])

    idx_3_0 = np.reshape(random.sample(range(0, 789), n_img), [-1, 1])
    idx_3_1 = np.reshape(random.sample(range(789, 2446), n_img), [-1, 1])
    idx_3_2 = np.reshape(random.sample(range(2446, np.size(x[3], 2)), n_img), [-1, 1])

    idx_4_0 = np.reshape(random.sample(range(0, 790), n_img), [-1, 1])
    idx_4_1 = np.reshape(random.sample(range(790, 2413), n_img), [-1, 1])
    idx_4_2 = np.reshape(random.sample(range(2413, np.size(x[4], 2)), n_img), [-1, 1])

    idx_5_0 = np.reshape(random.sample(range(0, 788), n_img), [-1, 1])
    idx_5_1 = np.reshape(random.sample(range(788, 1780), n_img), [-1, 1])
    idx_5_2 = np.reshape(random.sample(range(1780, np.size(x[4], 2)), n_img), [-1, 1])

    idx_6_0 = np.reshape(random.sample(range(0, 798), n_img), [-1, 1])
    idx_6_1 = np.reshape(random.sample(range(798, 2079), n_img), [-1, 1])
    idx_6_2 = np.reshape(random.sample(range(2079, np.size(x[4], 2)), n_img), [-1, 1])

    idx_7_0 = np.reshape(random.sample(range(0, 787), n_img), [-1, 1])
    idx_7_1 = np.reshape(random.sample(range(787, 2162), n_img), [-1, 1])
    idx_7_2 = np.reshape(random.sample(range(2162, np.size(x[4], 2)), n_img), [-1, 1])

    idx_8_0 = np.reshape(random.sample(range(0, 885), n_img), [-1, 1])
    idx_8_1 = np.reshape(random.sample(range(885, 2377), n_img), [-1, 1])
    idx_8_2 = np.reshape(random.sample(range(2377, np.size(x[4], 2)), n_img), [-1, 1])

    idx_9_0 = np.reshape(random.sample(range(0, 807), n_img), [-1, 1])
    idx_9_1 = np.reshape(random.sample(range(807, 2775), n_img), [-1, 1])
    idx_9_2 = np.reshape(random.sample(range(2775, np.size(x[4], 2)), n_img), [-1, 1])

    x_0_0 = np.reshape(x[0][:,:,idx_0_0], [130, 100, n_img])
    x_1_0 = np.reshape(x[1][:,:,idx_1_0], [130, 100, n_img])
    x_2_0 = np.reshape(x[2][:,:,idx_2_0], [130, 100, n_img])
    x_3_0 = np.reshape(x[3][:,:,idx_3_0], [130, 100, n_img])
    x_4_0 = np.reshape(x[4][:,:,idx_4_0], [130, 100, n_img])
    x_0_1 = np.reshape(x[0][:,:,idx_0_1], [130, 100, n_img])
    x_0_2 = np.reshape(x[0][:,:,idx_0_2], [130, 100, n_img])
    x_1_1 = np.reshape(x[1][:,:,idx_1_1], [130, 100, n_img])
    x_1_2 = np.reshape(x[1][:,:,idx_1_2], [130, 100, n_img])
    x_2_1 = np.reshape(x[2][:,:,idx_2_1], [130, 100, n_img])
    x_2_2 = np.reshape(x[2][:,:,idx_2_2], [130, 100, n_img])
    x_3_1 = np.reshape(x[3][:,:,idx_3_1], [130, 100, n_img])
    x_3_2 = np.reshape(x[3][:,:,idx_3_2], [130, 100, n_img])
    x_4_1 = np.reshape(x[4][:,:,idx_4_1], [130, 100, n_img])
    x_4_2 = np.reshape(x[4][:,:,idx_4_2], [130, 100, n_img])
    x_5_0 = np.reshape(x[4][:,:,idx_4_0], [130, 100, n_img])
    x_5_1 = np.reshape(x[4][:,:,idx_4_1], [130, 100, n_img])
    x_5_2 = np.reshape(x[4][:,:,idx_4_2], [130, 100, n_img])
    x_6_0 = np.reshape(x[4][:,:,idx_4_0], [130, 100, n_img])
    x_6_1 = np.reshape(x[4][:,:,idx_4_1], [130, 100, n_img])
    x_6_2 = np.reshape(x[4][:,:,idx_4_2], [130, 100, n_img])
    x_7_0 = np.reshape(x[4][:,:,idx_4_0], [130, 100, n_img])
    x_7_1 = np.reshape(x[4][:,:,idx_4_1], [130, 100, n_img])
    x_7_2 = np.reshape(x[4][:,:,idx_4_2], [130, 100, n_img])
    x_8_0 = np.reshape(x[4][:,:,idx_4_0], [130, 100, n_img])
    x_8_1 = np.reshape(x[4][:,:,idx_4_1], [130, 100, n_img])
    x_8_2 = np.reshape(x[4][:,:,idx_4_2], [130, 100, n_img])
    x_9_0 = np.reshape(x[4][:,:,idx_4_0], [130, 100, n_img])
    x_9_1 = np.reshape(x[4][:,:,idx_4_1], [130, 100, n_img])
    x_9_2 = np.reshape(x[4][:,:,idx_4_2], [130, 100, n_img])
    return x_0_0, x_0_1, x_0_2, x_1_0, x_1_1, x_1_2, x_2_0, x_2_1, x_2_2, x_3_0, x_3_1, x_3_2, x_4_0, x_4_1, x_4_2, \
           x_5_0, x_5_1, x_5_2, x_6_0, x_6_1, x_6_2, x_7_0, x_7_1, x_7_2, x_8_0, x_8_1, x_8_2, x_9_0, x_9_1, x_9_2


def main():
    x, y = preprocess()
    for this_ex in range(9):
        # tmp = x[this_ex] - np.mean(x[this_ex], 2)[:,:,None]
        x[this_ex] -= np.mean(x[this_ex], 2)[:,:,None]
    for this_choose in range(5):
        print('============================='+str(this_choose)+'=============================')
        x_0_0, x_0_1, x_0_2, x_1_0, x_1_1, x_1_2, x_2_0, x_2_1, x_2_2, x_3_0, x_3_1, x_3_2, x_4_0, x_4_1, x_4_2, x_5_0, \
        x_5_1, x_5_2, x_6_0, x_6_1, x_6_2, x_7_0, x_7_1, x_7_2, x_8_0, x_8_1, x_8_2, x_9_0, x_9_1, x_9_2 = choose_img(x)

        new_x = [np.concatenate((x_0_0, x_0_1, x_0_2), axis=2),
                 np.concatenate((x_1_0, x_1_1, x_1_2), axis=2),
                 np.concatenate((x_2_0, x_2_1, x_2_2), axis=2),
                 np.concatenate((x_3_0, x_3_1, x_3_2), axis=2),
                 np.concatenate((x_4_0, x_4_1, x_4_2), axis=2),
                 np.concatenate((x_5_0, x_5_1, x_5_2), axis=2),
                 np.concatenate((x_6_0, x_6_1, x_6_2), axis=2),
                 np.concatenate((x_7_0, x_7_1, x_7_2), axis=2),
                 np.concatenate((x_8_0, x_8_1, x_8_2), axis=2),
                 np.concatenate((x_9_0, x_9_1, x_9_2), axis=2),
                 ]

        for fold_j in range(0, 10):
            train_acc_save = np.zeros((1, 30000))
            test_acc_save = np.zeros((1, 30000))
            x_te = new_x[fold_j]
            x_te = np.transpose(x_te, [2, 0, 1])
            y_te = np.zeros((785*3, 3))
            y_te[0:785, 0] = 1
            y_te[785:2*785, 1] = 1
            y_te[2*785:3 * 785, 2] = 1
            idx = list(set(np.arange(10))-set(np.arange(fold_j, fold_j+1)))
            for k in range(9):
                this_x_tr = new_x[idx[k]]
                np.save('./data/new_all_same_size_face/0/'+str(k)+'xtr', this_x_tr)
                # this_y_tr = new_y[idx[k]]
                # np.save('./data/new_all_same_size_face/0/'+str(k)+'ytr', this_y_tr)

            x_tr_0 = np.load('./data/new_all_same_size_face/0/0xtr.npy')
            x_tr_1 = np.load('./data/new_all_same_size_face/0/1xtr.npy')
            x_tr_2 = np.load('./data/new_all_same_size_face/0/2xtr.npy')
            x_tr_3 = np.load('./data/new_all_same_size_face/0/3xtr.npy')
            x_tr_4 = np.load('./data/new_all_same_size_face/0/4xtr.npy')
            x_tr_5 = np.load('./data/new_all_same_size_face/0/5xtr.npy')
            x_tr_6 = np.load('./data/new_all_same_size_face/0/6xtr.npy')
            x_tr_7 = np.load('./data/new_all_same_size_face/0/7xtr.npy')
            x_tr_8 = np.load('./data/new_all_same_size_face/0/8xtr.npy')
            x_tr = np.concatenate((x_tr_0, x_tr_1, x_tr_2, x_tr_3, x_tr_4, x_tr_5, x_tr_6, x_tr_7, x_tr_8), axis=2)

            model = get_model(x_in)

            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=y_in), 0)
            # training with Adam
            train_step = tf.train.AdamOptimizer().minimize(loss)

            correct_pred = tf.equal(tf.argmax(model, 1), tf.argmax(y_in, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), 0)
            # saver = tf.train.Saver()

            y_test_new = np.zeros((3, 3))
            y_test_new[0, 0] = 1
            y_test_new[1, 1] = 1
            y_test_new[2, 2] = 1
            with tf.device('/gpu:0'):
                with tf.Session() as sess:
                    sess.run(tf.global_variables_initializer())
                    for i in range(0, 10000):
                        index = i % 785
                        person_1_label_1_train_begin = index * 1
                        person_1_label_1_train_end = (index + 1) * 1
                        person_1_label_2_train_begin = index * 1 + 785
                        person_1_label_2_train_end = (index + 1) * 1 + 785
                        person_1_label_3_train_begin = index * 1 + 785 * 2
                        person_1_label_3_train_end = (index + 1) * 1 + 785 * 2

                        person_2_label_1_train_begin = index * 1 + 785 * 3
                        person_2_label_1_train_end = (index + 1) * 1 + 785 * 3
                        person_2_label_2_train_begin = index * 1 + 785 * 4
                        person_2_label_2_train_end = (index + 1) * 1 + 785 * 4
                        person_2_label_3_train_begin = index * 1 + 785 * 5
                        person_2_label_3_train_end = (index + 1) * 1 + 785 * 5

                        person_3_label_1_train_begin = index * 1 + 785 * 6
                        person_3_label_1_train_end = (index + 1) * 1 + 785 * 6
                        person_3_label_2_train_begin = index * 1 + 785 * 7
                        person_3_label_2_train_end = (index + 1) * 1 + 785 * 7
                        person_3_label_3_train_begin = index * 1 + 785 * 8
                        person_3_label_3_train_end = (index + 1) * 1 + 785 * 8

                        person_4_label_1_train_begin = index * 1 + 785 * 9
                        person_4_label_1_train_end = (index + 1) * 1 + 785 * 9
                        person_4_label_2_train_begin = index * 1 + 785 * 10
                        person_4_label_2_train_end = (index + 1) * 1 + 785 * 10
                        person_4_label_3_train_begin = index * 1 + 785 * 11
                        person_4_label_3_train_end = (index + 1) * 1 + 785 * 11

                        person_5_label_1_train_begin = index * 1 + 785 * 12
                        person_5_label_1_train_end = (index + 1) * 1 + 785 * 12
                        person_5_label_2_train_begin = index * 1 + 785 * 13
                        person_5_label_2_train_end = (index + 1) * 1 + 785 * 13
                        person_5_label_3_train_begin = index * 1 + 785 * 14
                        person_5_label_3_train_end = (index + 1) * 1 + 785 * 14

                        person_6_label_1_train_begin = index * 1 + 785 * 15
                        person_6_label_1_train_end = (index + 1) * 1 + 785 * 15
                        person_6_label_2_train_begin = index * 1 + 785 * 16
                        person_6_label_2_train_end = (index + 1) * 1 + 785 * 16
                        person_6_label_3_train_begin = index * 1 + 785 * 17
                        person_6_label_3_train_end = (index + 1) * 1 + 785 * 17

                        person_7_label_1_train_begin = index * 1 + 785 * 18
                        person_7_label_1_train_end = (index + 1) * 1 + 785 * 18
                        person_7_label_2_train_begin = index * 1 + 785 * 19
                        person_7_label_2_train_end = (index + 1) * 1 + 785 * 19
                        person_7_label_3_train_begin = index * 1 + 785 * 20
                        person_7_label_3_train_end = (index + 1) * 1 + 785 * 20

                        person_8_label_1_train_begin = index * 1 + 785 * 21
                        person_8_label_1_train_end = (index + 1) * 1 + 785 * 21
                        person_8_label_2_train_begin = index * 1 + 785 * 22
                        person_8_label_2_train_end = (index + 1) * 1 + 785 * 22
                        person_8_label_3_train_begin = index * 1 + 785 * 23
                        person_8_label_3_train_end = (index + 1) * 1 + 785 * 23

                        person_9_label_1_train_begin = index * 1 + 785 * 24
                        person_9_label_1_train_end = (index + 1) * 1 + 785 * 24
                        person_9_label_2_train_begin = index * 1 + 785 * 25
                        person_9_label_2_train_end = (index + 1) * 1 + 785 * 25
                        person_9_label_3_train_begin = index * 1 + 785 * 26
                        person_9_label_3_train_end = (index + 1) * 1 + 785 * 26

                        this_x_train = np.concatenate((x_tr[:, :, person_1_label_1_train_begin:person_1_label_1_train_end],
                                                      x_tr[:, :, person_1_label_2_train_begin:person_1_label_2_train_end],
                                                      x_tr[:, :, person_1_label_3_train_begin:person_1_label_3_train_end],
                                                      x_tr[:, :, person_2_label_1_train_begin:person_2_label_1_train_end],
                                                      x_tr[:, :, person_2_label_2_train_begin:person_2_label_2_train_end],
                                                      x_tr[:, :, person_2_label_3_train_begin:person_2_label_3_train_end],
                                                      x_tr[:, :, person_3_label_1_train_begin:person_3_label_1_train_end],
                                                      x_tr[:, :, person_3_label_2_train_begin:person_3_label_2_train_end],
                                                      x_tr[:, :, person_3_label_3_train_begin:person_3_label_3_train_end],
                                                      x_tr[:, :, person_4_label_1_train_begin:person_4_label_1_train_end],
                                                      x_tr[:, :, person_4_label_2_train_begin:person_4_label_2_train_end],
                                                      x_tr[:, :, person_4_label_3_train_begin:person_4_label_3_train_end],
                                                      x_tr[:, :, person_5_label_1_train_begin:person_5_label_1_train_end],
                                                      x_tr[:, :, person_5_label_2_train_begin:person_5_label_2_train_end],
                                                      x_tr[:, :, person_5_label_3_train_begin:person_5_label_3_train_end],
                                                      x_tr[:, :, person_6_label_1_train_begin:person_6_label_1_train_end],
                                                      x_tr[:, :, person_6_label_2_train_begin:person_6_label_2_train_end],
                                                      x_tr[:, :, person_6_label_3_train_begin:person_6_label_3_train_end],
                                                      x_tr[:, :, person_7_label_1_train_begin:person_7_label_1_train_end],
                                                      x_tr[:, :, person_7_label_2_train_begin:person_7_label_2_train_end],
                                                      x_tr[:, :, person_7_label_3_train_begin:person_7_label_3_train_end],
                                                      x_tr[:, :, person_8_label_1_train_begin:person_8_label_1_train_end],
                                                      x_tr[:, :, person_8_label_2_train_begin:person_8_label_2_train_end],
                                                      x_tr[:, :, person_8_label_3_train_begin:person_8_label_3_train_end],
                                                       x_tr[:, :, person_9_label_1_train_begin:person_9_label_1_train_end],
                                                       x_tr[:, :, person_9_label_2_train_begin:person_9_label_2_train_end],
                                                       x_tr[:, :, person_9_label_3_train_begin:person_9_label_3_train_end]), axis=2)
                                                      # x_tr[:, :, person_5_label_1_train_begin:person_5_label_1_train_end],
                                                      # x_tr[:, :, person_5_label_2_train_begin:person_5_label_2_train_end],
                                                      # x_tr[:, :, person_5_label_3_train_begin:person_5_label_3_train_end],
                                                      # x_tr[:, :, person_6_label_1_train_begin:person_6_label_1_train_end],
                                                      # x_tr[:, :, person_6_label_2_train_begin:person_6_label_2_train_end],
                                                      # x_tr[:, :, person_6_label_3_train_begin:person_6_label_3_train_end],
                                                      # x_tr[:, :, person_7_label_1_train_begin:person_7_label_1_train_end],
                                                      # x_tr[:, :, person_7_label_2_train_begin:person_7_label_2_train_end],
                                                      # x_tr[:, :, person_7_label_3_train_begin:person_7_label_3_train_end],
                                                      # x_tr[:, :, person_8_label_1_train_begin:person_8_label_1_train_end],
                                                      # x_tr[:, :, person_8_label_2_train_begin:person_8_label_2_train_end],
                                                      # x_tr[:, :, person_8_label_3_train_begin:person_8_label_3_train_end],


                        this_x_train = np.transpose(this_x_train, [2, 0, 1])
                        this_y_train = np.zeros((27, 3))
                        for this_tmp in range(27):
                            # this_y_train[int(this_tmp * 1):int((this_tmp + 1) * n_img), int(this_tmp % 3)] = 1
                            this_y_train[int(this_tmp), int(this_tmp%3)] = 1
                        _, = sess.run([train_step], feed_dict={x_in: this_x_train, y_in: this_y_train})
                        # _, lo, mo = sess.run([train_step, loss, model], feed_dict={x_in: this_x_train, y_in: this_y_train})
                        if i % 20 == 0:
                            # print("train time: ", str(stop-start))
                            this_train_loss = sess.run(loss, feed_dict={x_in: this_x_train, y_in: this_y_train})
                            this_train_accuracy = sess.run(accuracy, feed_dict={x_in: this_x_train, y_in: this_y_train})
                            acc_test_t = 0
                            loss_test_t = 0
                            for i_test in range(785):
                                this_test_batch = np.zeros((3, 130, 100))
                                this_test_batch[0, :, :] = x_te[i_test]
                                this_test_batch[1, :, :] = x_te[int(i_test + 785)]
                                this_test_batch[2, :, :] = x_te[int(i_test + 785*2)]

                                # this_test_loss = sess.run(loss, feed_dict={x_in:x_te, y_in: y_te})
                                # this_test_accuracy = sess.run(accuracy, feed_dict={x_in: x_te, y_in: y_te})

                                this_test_loss = sess.run(loss, feed_dict={x_in:this_test_batch, y_in: y_test_new})
                                this_test_accuracy = sess.run(accuracy, feed_dict={x_in: this_test_batch, y_in: y_test_new})
                                loss_test_t += this_test_loss
                                acc_test_t += this_test_accuracy

                            acc_test = acc_test_t/785
                            loss_test = loss_test_t/785
                            train_acc_save[0,i] = acc_test
                            test_acc_save[0,i] = this_train_accuracy

                            print('The', i, 'time train accuracy: ', this_train_accuracy)
                            print('The', i, 'time train loss: ', this_train_loss, ', train accuracy: ', this_train_accuracy)
                            print('The', i, 'time test loss: ', loss_test, ', test accuracy: ', acc_test)
            np.save('./data/'+str(fold_j)+'train_acc', train_acc_save)
            np.save('./data/'+str(fold_j)+'test_acc', test_acc_save)
            print('saved')


if __name__ == '__main__':
    main()