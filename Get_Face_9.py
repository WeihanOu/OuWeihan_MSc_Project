import numpy as np
from ReadImg import *
from ExtractFeature import *
import random
from sklearn.svm import LinearSVC
import cv2
from scipy.misc import imresize

file_lin = './9.dat'


def prepare_data(img_frames, time_arr):
    # split train and test set
    time_arr = time_arr[1:]
    relex_begin = np.amin(np.where(time_arr > 3))
    relex_over = np.amin(np.where(time_arr > 93))
    stress_1_1_begin = np.amin(np.where(time_arr > 112))
    stress_1_1_over = np.amin(np.where(time_arr > 178))
    stress_1_2_begin = np.amin(np.where(time_arr > 267))
    stress_1_2_over = np.amin(np.where(time_arr > 314))
    stress_2_1_begin = np.amin(np.where(time_arr > 365))
    stress_2_1_over = np.amin(np.where(time_arr > 414))
    stress_2_2_begin = np.amin(np.where(time_arr > 546))
    stress_2_2_over = np.amin(np.where(time_arr > 596))
    relax = relex_over-50-relex_begin
    stress1 = stress_1_1_over-stress_1_1_begin+stress_1_2_over-50-stress_1_2_begin
    stress2 = stress_2_1_over-stress_2_1_begin+stress_2_2_over-50-stress_2_2_begin
    """" new version do not distinguish train /test"""
    x_train = np.concatenate((img_frames[:, :, relex_begin:relex_over],
                              img_frames[:, :, stress_1_1_begin:stress_1_1_over],
                              img_frames[:, :, stress_1_2_begin:stress_1_2_over],
                              img_frames[:, :, stress_2_1_begin:stress_2_1_over],
                              img_frames[:, :, stress_2_2_begin:stress_2_2_over]), axis=2)

    """" old version distinguish train/test for individual analysis"""
    """new version"""
    y_train = np.zeros(np.size(x_train, 2))
    y_train[0:relex_over-relex_begin] = 0
    y_train[(relex_over-relex_begin):(relex_over-relex_begin) + (stress_1_1_over-stress_1_1_begin) + (stress_1_2_over-stress_1_2_begin)] = 1
    y_train[(relex_over-relex_begin) + (stress_1_1_over-stress_1_1_begin) + (stress_1_2_over-stress_1_2_begin):
            (relex_over-relex_begin)+(stress_1_1_over-stress_1_1_begin)+(stress_1_2_over-stress_1_2_begin)+
            (stress_2_1_over - stress_2_1_begin) + (stress_2_2_over - stress_2_2_begin)] = 2
    return x_train, y_train


def extract_face(x_train_ori):
    for i in range(np.size(x_train_ori, 2)):
        x_train_ori[:, :, i] = np.rot90(x_train_ori[:, :, i], 2)
    # x_train_ori -= np.mean(x_train_ori, 2)[:,:,None]
    x_train_mean = np.mean(x_train_ori, 2)
    x_train = x_train_ori[0:152, :, :]
    # for i in range(np.size(x_train, 2)):
    #     plt.imsave('./data/faces/yj/train_' + str(i) + '.png', x_train[:, :, i])
    #     first = x_train[:, :, 0]
    #     last = x_train[:, :, 1647]
    # x_test = x_test[0:204, :, :]
    x_train_mean = np.mean(x_train, 2)
    # first = x_train[:, :, 0]
    # last = x_train[:, :, 2646]
    # plt.imsave('./data/faces/yj/___0000.png', first)
    # plt.imsave('./data/faces/yj/___2646.png', last)
    # get four lines to cut the face from data_frames
    x_train_mean_2d = np.reshape(x_train_mean, (152*240))
    top = int(np.floor(np.amin(np.where(x_train_mean_2d > 27.5))/240))
    bottom = int(np.floor(np.amax(np.where(x_train_mean_2d > 28)) / 240))
    x_train_mean_2d = np.reshape(np.transpose(x_train_mean), (152 * 240))
    left = int(np.floor(np.amin(np.where(x_train_mean_2d > 27)) / 152))
    right = int(np.floor(np.amax(np.where(x_train_mean_2d > 26.5)) / 152))
    x_train_mean_cut = x_train_mean[top:bottom, left:right]
    # plt.imsave('./data/faces/yj/face_mean.png', x_train_mean_cut)
    face_hight = bottom-top
    face_width = right-left
    # cut train data_frames
    x_train_cut = np.zeros((face_hight,face_width, np.size(x_train, 2)))
    test = x_train[:,:,-1]
    # plt.imsave('./data/faces/yj/000' + str(0) + '.png', test)
    for i in range(np.size(x_train, 2)):
        this_face = np.reshape(x_train[:, :, i], (152 * 240))
        this_top = int(np.floor(np.amin(np.where(this_face > 27.5)) / 240))
        this_face = np.reshape(np.transpose(x_train[:, :, i]), (152 * 240))
        this_left = int(np.floor(np.amin(np.where(this_face > 27)) / 152))
        x_train_cut[:, :, i] = x_train_ori[top:bottom, this_left:this_left+face_width, i]
        if i < 1780:
            x_train_cut[:, :, i] = cv2.resize(x_train_cut[7:face_hight, :, i],(face_width, face_hight), interpolation=cv2.INTER_CUBIC)
        # plt.imsave('./data/faces/yj/train_' + str(i) + '.png', x_train_cut[:, :, i])
    return x_train_cut, abs(top-bottom), (right-left)


def get_face_9():
    # read img & prepossessing
    file = read_file(file_lin, 320, 240)
    img_frames = get_img_frames(file, 320, 240)
    # prepare face_data
    time_arr = get_time_arr(file)  #246.149s
    """ old version"""
    # x_train, x_test, y_train, y_test = prepare_data(img_frames, time_arr)
    """ new version"""
    x_train, y_train = prepare_data(img_frames, time_arr)
    # face_train, face_test, h, w = extract_face(x_train, x_test)
    face_train, h, w = extract_face(x_train)

    return face_train, y_train, h, w
if __name__ == '__main__':
    get_face_9()



