import numpy as np
from ReadImg import *
from ExtractFeature import *
import random
from sklearn.svm import LinearSVC


file_lin = './8.dat'


def prepare_data(img_frames, time_arr):
    # split train and test set
    time_arr = time_arr[1:]
    relex_begin = np.amin(np.where(time_arr > 20))
    relex_over = np.amin(np.where(time_arr > 121))
    stress_1_1_begin = np.amin(np.where(time_arr > 185))
    stress_1_1_over = np.amin(np.where(time_arr > 259))
    stress_1_2_begin = np.amin(np.where(time_arr > 315))
    stress_1_2_over = np.amin(np.where(time_arr > 411))
    stress_2_1_begin = np.amin(np.where(time_arr > 535))
    stress_2_1_over = np.amin(np.where(time_arr > 610))
    stress_2_2_begin = np.amin(np.where(time_arr > 640))
    stress_2_2_over = np.amin(np.where(time_arr > 700))
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


def extract_face(x_train):
    for i in range(np.size(x_train, 2)):
        x_train[:, :, i] = np.rot90(x_train[:, :, i], 2)
    x_train_mean = np.mean(x_train, 2)
    # plt.imsave('./data/faces/ou/x_mean.png', x_train_mean)
    x_train = x_train[0:186, :, :]
    # x_test = x_test[0:204, :, :]
    x_train_mean = np.mean(x_train, 2)
    # plt.imsave('./data/faces/ou/face_mean.png', x_train_mean)
    # get four lines to cut the face from data_frames
    x_train_mean_2d = np.reshape(x_train_mean, (186*240))
    top = int(np.floor(np.amin(np.where(x_train_mean_2d > 29.4))/240))
    bottom = int(np.floor(np.amax(np.where(x_train_mean_2d > 29.5)) / 240))
    x_train_mean_2d = np.reshape(np.transpose(x_train_mean), (186 * 240))
    left = int(np.floor(np.amin(np.where(x_train_mean_2d > 29.5)) / 186))
    right = int(np.floor(np.amax(np.where(x_train_mean_2d > 29.4)) / 186))
    x_train_mean_cut = x_train_mean[top:bottom, left:right]
    # plt.imsave('./data/faces/ou/face_mean.png', x_train_mean_cut)
    face_hight = bottom-top
    face_width = right-left+5
    # cut train data_frames
    x_train_cut = np.zeros((face_hight,face_width, np.size(x_train, 2)))
    # test = x_train[:,:,-1]
    # plt.imsave('./data/faces/ou/000' + str(0) + '.png', test)
    for i in range(np.size(x_train, 2)):
        this_face = np.reshape(np.transpose(x_train[:, :, i]), (186 * 240))
        this_top = int(np.floor(np.amin(np.where(this_face > 29.4)) / 240))
        this_left = int(np.floor(np.amin(np.where(this_face > 29.5)) / 186))
        x_train_cut[:, :, i] = x_train[top:bottom, this_left:this_left+face_width, i]
        # plt.imsave('./data/faces/ou/' + str(i) + '.png', x_train_cut[:, :, i])
    return x_train_cut, abs(top-bottom), (right-left)


def get_face_8():
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
    get_face_8()



