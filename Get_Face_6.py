import numpy as np
from ReadImg import *
from ExtractFeature import *
import random
from sklearn.svm import LinearSVC


file_mao = 'D:/PycharmWorkSpace/MScProj/data/021-mao.dat'


def prepare_data(img_frames, time_arr):
    # split train and test set
    time_arr = time_arr[1:]
    relex_begin = np.amin(np.where(time_arr > 30))
    relex_re_b = np.amin(np.where(time_arr > 65))
    relex_re_e = np.amin(np.where(time_arr > 70))
    relex_over = np.amin(np.where(time_arr > 130))
    stress_1_1_begin = np.amin(np.where(time_arr > 170))
    stress_1_1_over = np.amin(np.where(time_arr > 265))
    stress_1_2_begin = np.amin(np.where(time_arr > 296))
    stress_1_2_over = np.amin(np.where(time_arr > 390))
    stress_2_1_begin = np.amin(np.where(time_arr > 452))
    s21_re_b = np.amin(np.where(time_arr > 480))
    s21_re_e = np.amin(np.where(time_arr > 500))
    stress_2_1_over = np.amin(np.where(time_arr > 538))
    stress_2_2_begin = np.amin(np.where(time_arr > 572))
    stress_2_2_over = np.amin(np.where(time_arr > 656))
    relax = relex_over-50-relex_re_e+relex_re_b-relex_begin
    stress1 = stress_1_1_over-stress_1_1_begin+stress_1_2_over-50-s21_re_e+s21_re_b-stress_1_2_begin
    stress2 = stress_2_1_over-stress_2_1_begin+stress_2_2_over-50-stress_2_2_begin
    # x_train = np.concatenate((img_frames[:, :, relex_begin:relex_re_b],
    #                           img_frames[:, :, relex_re_e:relex_over-50],
    #                           img_frames[:, :, stress_1_1_begin:stress_1_1_over],
    #                           img_frames[:, :, stress_1_2_begin:stress_1_2_over-50],
    #                           img_frames[:, :, stress_2_1_begin:s21_re_b],
    #                           img_frames[:, :, s21_re_e:stress_2_1_over],
    #                           img_frames[:, :, stress_2_2_begin:stress_2_2_over-50]), axis=2)
    x_train = np.concatenate((img_frames[:, :, relex_begin:relex_re_b],
                              img_frames[:, :, relex_re_e:relex_over],
                              img_frames[:, :, stress_1_1_begin:stress_1_1_over],
                              img_frames[:, :, stress_1_2_begin:stress_1_2_over],
                              img_frames[:, :, stress_2_1_begin:s21_re_b],
                              img_frames[:, :, s21_re_e:stress_2_1_over],
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
    y_train[0:relex_over-relex_re_e+relex_re_b-relex_begin] = 0
    y_train[relex_over-relex_re_e+relex_re_b-relex_begin:
            relex_over-relex_re_e+relex_re_b-relex_begin +
            (stress_1_1_over-s21_re_e+s21_re_b-stress_1_1_begin) +
            (stress_1_2_over-stress_1_2_begin)] = 1
    y_train[relex_over-relex_re_e+relex_re_b-relex_begin +
            (stress_1_1_over-s21_re_e+s21_re_b-stress_1_1_begin) +
            (stress_1_2_over-stress_1_2_begin):
            relex_over - relex_re_e + relex_re_b - relex_begin +
            (stress_1_1_over - s21_re_e + s21_re_b - stress_1_1_begin) +
            (stress_1_2_over - stress_1_2_begin)+
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
    # cut no-use portion & calculate mean
    # x_train = x_train[15:, :, :]
    # x_test = x_test[15:, :, :]
    x_train_mean = np.mean(x_train, 2)
    plt.imsave('./data/faces/mao/x_mean.png', x_train_mean)
    x_train = x_train[0:178, :, :]
    x_train_mean = np.mean(x_train, 2)
    plt.imsave('./data/faces/mao/face_mean.png', x_train_mean)
    # get four lines to cut the face from data_frames
    x_train_mean_2d = np.reshape(x_train_mean, (178*240))
    top = int(np.floor(np.amin(np.where(x_train_mean_2d > 28))/240))
    bottom = int(np.floor(np.amax(np.where(x_train_mean_2d > 28)) / 240))
    x_train_mean_2d = np.reshape(np.transpose(x_train_mean), (178 * 240))
    left = int(np.floor(np.amin(np.where(x_train_mean_2d > 27)) / 178))
    right = int(np.floor(np.amax(np.where(x_train_mean_2d > 27)) / 178))
    x_train_mean_cut = x_train_mean[top:bottom, left:right]
    plt.imsave('./data/faces/mao/face_mean.png', x_train_mean_cut)
    # cut train data_frames
    x_train_cut = x_train[top:bottom, left:right, :]
    # for i_save in range(np.size(x_train, 2)):
    #     this_frame = x_train_cut[:, :, i_save]
    #     plt.imsave('./data/faces/mao/train_' + str(i_save) + '.png', this_frame)
    # cut test data_frames
    # x_test_cut = x_test[top:bottom, left:right, :]
    # for i_save in range(np.size(x_test, 2)):
    #     this_frame = x_test_cut[:, :, i_save]
    #     # plt.imsave('./data/faces/mao/test_' + str(i_save) + '.png', this_frame)
    return x_train_cut, abs(top-bottom), (right-left)


def get_face_mao():
    # read img & prepossessing
    file = read_file(file_mao, 320, 240)
    img_frames = get_img_frames(file, 320, 240)
    # prepare face_data
    time_arr = get_time_arr(file)  #246.149s
    x_train, y_train = prepare_data(img_frames, time_arr)
    # x_train, x_test, y_train, y_test = prepare_data(img_frames, time_arr)
    face_train, h, w = extract_face(x_train)
    # face_train, face_test, h, w = extract_face(x_train, x_test)
    # img_frames = img_frames/np.mean(img_frames)
    # return face_train, face_test,  y_train, y_test, h, w
    # eigen_faces, x_train_pca, x_test_pca, var_percent = pca(face_train, face_test, h, w)
    # for i_save in range(np.size(eigen_faces, 0)):
    #     this_eg = eigen_faces[i_save, :]
    #     plt.imsave('./data/eigen/mao/' + str(i_save) + '.png', (np.reshape(this_eg, [h, w])))
    # print(var_percent)
    # print(sum(var_percent))
    # classifier = LinearSVC()
    # classifier.fit(x_train_pca, y_train)
    # train_accuracy = classifier.score(x_train_pca, y_train)
    # test_accuracy = classifier.score(x_test_pca, y_test)
    # print(train_accuracy)
    # print(test_accuracy)
    return face_train, y_train, h, w
    # return face_train, face_test, y_train, y_test, h, w

if __name__ == '__main__':
    get_face_mao()



