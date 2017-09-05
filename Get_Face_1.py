import numpy as np
from ReadImg import *
from ExtractFeature import *
import random
from sklearn.svm import LinearSVC
import cv2

file_ch = 'D:/PycharmWorkSpace/MScProj/data/201-ch.dat'


def prepare_data(img_frames, time_arr):
    # split train and test set
    time_arr = time_arr[1:]
    relex_begin = np.amin(np.where(time_arr > 3))
    relex_over = np.amin(np.where(time_arr > 95))
    stress_1_1_begin = np.amin(np.where(time_arr > 160))
    stress_1_1_over = np.amin(np.where(time_arr > 240))
    stress_1_2_begin = np.amin(np.where(time_arr > 305))
    stress_1_2_over = np.amin(np.where(time_arr > 398))
    stress_2_1_begin = np.amin(np.where(time_arr > 432))
    stress_2_1_over = np.amin(np.where(time_arr > 498))
    stress_2_2_begin = np.amin(np.where(time_arr > 530))
    stress_2_2_over = np.amin(np.where(time_arr > 609))
    relax = relex_over - 50 - relex_begin
    stress1 = stress_1_1_over - stress_1_1_begin + stress_1_2_over - 50 - stress_1_2_begin
    stress2 = stress_2_1_over - stress_2_1_begin + stress_2_2_over - 50 - stress_2_2_begin
    # x_train = np.concatenate((img_frames[:, :, relex_begin:relex_over-50],
    #                           img_frames[:, :, stress_1_1_begin:stress_1_1_over],
    #                           img_frames[:, :, stress_1_2_begin:stress_1_2_over-50],
    #                           img_frames[:, :, stress_2_1_begin:stress_2_1_over],
    #                           img_frames[:, :, stress_2_2_begin:stress_2_2_over-50]), axis=2)
    x_train = np.concatenate((img_frames[:, :, relex_begin:relex_over],
                              img_frames[:, :, stress_1_1_begin:stress_1_1_over],
                              img_frames[:, :, stress_1_2_begin:stress_1_2_over],
                              img_frames[:, :, stress_2_1_begin:stress_2_1_over],
                              img_frames[:, :, stress_2_2_begin:stress_2_2_over]), axis=2)
    # x_test = np.concatenate((img_frames[:, :, relex_over-50:relex_over],
    #                          img_frames[:, :, stress_1_2_over-50:stress_1_2_over],
    #                          img_frames[:, :, stress_2_2_over-50:stress_2_2_over]), axis=2)
    # for i_save in range(np.size(x_test, 2)):
    #     this_frame = x_test[:, :, i_save]
    #     plt.imsave('./data/test_ou_' + str(i_save) + '.png', this_frame)
    # apply labels to train & test set
    # y_train = np.zeros(np.size(x_train, 2))
    # y_train[0:relex_over-50-relex_re_e+relex_re_b-relex_begin] = 0
    # y_train[relex_over-50-relex_re_e+relex_re_b-relex_begin:
    #         relex_over-50-relex_re_e+relex_re_b-relex_begin +
    #         (stress_1_1_over-s21_re_e+s21_re_b-stress_1_1_begin) +
    #         (stress_1_2_over-50-stress_1_2_begin)] = 1
    # y_train[relex_over-50-relex_re_e+relex_re_b-relex_begin +
    #         (stress_1_1_over-s21_re_e+s21_re_b-stress_1_1_begin) +
    #         (stress_1_2_over-50-stress_1_2_begin):
    #         relex_over - 50 - relex_re_e + relex_re_b - relex_begin +
    #         (stress_1_1_over - s21_re_e + s21_re_b - stress_1_1_begin) +
    #         (stress_1_2_over - 50 - stress_1_2_begin)+
    #         (stress_2_1_over - stress_2_1_begin) + (stress_2_2_over - 50 - stress_2_2_begin)] = 2
    """new version"""
    y_train = np.zeros(np.size(x_train, 2))
    y_train[0:relex_over - relex_begin] = 0
    y_train[(relex_over - relex_begin):(relex_over - relex_begin) + (stress_1_1_over - stress_1_1_begin) + (
    stress_1_2_over - stress_1_2_begin)] = 1
    y_train[(relex_over - relex_begin) + (stress_1_1_over - stress_1_1_begin) + (stress_1_2_over - stress_1_2_begin):
    (relex_over - relex_begin) + (stress_1_1_over - stress_1_1_begin) + (stress_1_2_over - stress_1_2_begin) +
    (stress_2_1_over - stress_2_1_begin) + (stress_2_2_over - stress_2_2_begin)] = 2
    # y_test = np.zeros(150)
    # y_test[0:50] = 0
    # y_test[50:100] = 1
    # y_test[100:150] = 2
    # return x_train, x_test, y_train, y_test
    return x_train, y_train


# def extract_face(x_train, x_test):
#     # cut no-use portion & calculate mean
#     # x_train = x_train[15:, :, :]
#     # x_test = x_test[15:, :, :]
#     x_train_mean = np.mean(x_train, 2)
#     plt.imsave('./data/faces/mao/x_mean.png', x_train_mean)
#     x_train = x_train[0:178, :, :]
#     x_train_mean = np.mean(x_train, 2)
#     plt.imsave('./data/faces/mao/face_mean.png', x_train_mean)
#     # get four lines to cut the face from data_frames
#     x_train_mean_2d = np.reshape(x_train_mean, (178*240))
#     top = int(np.floor(np.amin(np.where(x_train_mean_2d > 28))/240))
#     bottom = int(np.floor(np.amax(np.where(x_train_mean_2d > 28)) / 240))
#     x_train_mean_2d = np.reshape(np.transpose(x_train_mean), (178 * 240))
#     left = int(np.floor(np.amin(np.where(x_train_mean_2d > 27)) / 178))
#     right = int(np.floor(np.amax(np.where(x_train_mean_2d > 27)) / 178))
#     x_train_mean_cut = x_train_mean[top:bottom, left:right]
#     plt.imsave('./data/faces/mao/face_mean.png', x_train_mean_cut)
#     # cut train data_frames
#     x_train_cut = x_train[top:bottom, left:right, :]
#     # for i_save in range(np.size(x_train, 2)):
#     #     this_frame = x_train_cut[:, :, i_save]
#     #     plt.imsave('./data/faces/mao/train_' + str(i_save) + '.png', this_frame)
#     # cut test data_frames
#     x_test_cut = x_test[top:bottom, left:right, :]
#     # for i_save in range(np.size(x_test, 2)):
#     #     this_frame = x_test_cut[:, :, i_save]
#     #     # plt.imsave('./data/faces/mao/test_' + str(i_save) + '.png', this_frame)
#     return x_train_cut, x_test_cut, abs(top-bottom), (right-left)


"""new version"""
def extract_face(x_train):
    for i in range(np.size(x_train, 2)):
        x_train[:, :, i] = np.rot90(x_train[:, :, i], 2)
    x_train_mean = np.mean(x_train, 2)
    plt.imsave('./data/faces/ch/x_mean.png', x_train_mean)
    x_train = x_train[0:176, :, :]
    # x_test = x_test[0:204, :, :]
    x_train_mean = np.mean(x_train, 2)
    plt.imsave('./data/faces/ch/face_mean.png', x_train_mean)
    # get four lines to cut the face from data_frames
    x_train_mean_2d = np.reshape(x_train_mean, (176*240))
    top = int(np.floor(np.amin(np.where(x_train_mean_2d > 35.8))/240))
    bottom = int(np.floor(np.amax(np.where(x_train_mean_2d > 36.8)) / 240))
    lr_x_train_mean = x_train_mean[0:150,:]
    lr_x_train_mean_2d = np.reshape(np.transpose(lr_x_train_mean), (150 * 240))
    left = int(np.floor(np.amin(np.where(lr_x_train_mean_2d[0:37200] > 35.5)) / 150))
    right = int(np.floor(np.amax(np.where(lr_x_train_mean_2d[0:37200] > 35.5)) / 150))
    x_train_mean_cut = x_train_mean[top:bottom, left:right]
    plt.imsave('./data/faces/ch/face_mean.png', x_train_mean_cut)
    t = x_train[:,:,3597]
    plt.imsave('./data/faces/ch/face_mean.png', t)
    # cut train data_frames
    face_hight = bottom-top
    face_width = right-left
    x_train_cut = np.zeros((face_hight,face_width-2, np.size(x_train, 2)))
    # x_train_cut = x_train[top:bottom, left:right, :]

    for i_save in range(np.size(x_train, 2)):
        this_face = np.reshape(x_train[:, :, i_save], (176 * 240))
        this_top = int(np.floor(np.amin(np.where(this_face > 35.8)) / 240))
        this_bottom = int(np.floor(np.amax(np.where(this_face > 36.8)) / 240))
        this_face_lr = np.reshape(np.transpose(x_train[70:120, :, i_save]),(50*240))
        this_left = int(np.floor(np.amin(np.where(this_face_lr > 35.5))/50))
        if this_left <76:
            this_left = 76
        # if i_save > 3550:
        #     test = np.where(this_face_lr > 35.5)
        #     test1 = np.amin(test)
        #     test2 = test1
        #     print(this_left)
            # print(this_face_lr[this_left*50])
        x_train_cut[:, :, i_save] =cv2.resize(x_train[this_top:this_bottom, this_left:this_left + face_width-2, i_save], (face_width-2, face_hight), interpolation=cv2.INTER_CUBIC)
        plt.imsave('./data/faces/ch/train_' + str(i_save) + '.png', x_train_cut[:, :, i_save])
    # cut test data_frames
    # x_test_cut = x_test[top:bottom, left:right, :]
    # for i_save in range(np.size(x_test, 2)):
    #     this_frame = x_test_cut[:, :, i_save]
    #     # plt.imsave('./data/faces/mao/test_' + str(i_save) + '.png', this_frame)
    return x_train_cut, abs(top-bottom), (right-left)


def get_face_ch():
    # read img & prepossessing
    file = read_file(file_ch, 320, 240)
    img_frames = get_img_frames(file, 320, 240)
    # prepare face_data
    time_arr = get_time_arr(file)  #246.149s
    x_train, y_train = prepare_data(img_frames, time_arr)
    face_train, h, w = extract_face(x_train)
    return face_train, y_train, h, w
    # return face_train, face_test, y_train, y_test, h, w

if __name__ == '__main__':
    get_face_ch()



