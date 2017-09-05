from Get_Face_Kun import *
from Get_Face_Lin import *
from Get_Face_Mao import *
from Get_Face_Nadia import *
from Th_Get_Face_Ma import *
from Get_Face_girl import *
from Get_Face_juleir import *
from Get_Face_Ou import *
from Get_Face_Youngjun import *
from Get_Face_ch import *
from PIL import Image
import cv2
base_w = 100
base_h = 130


def load_faces():
    nadia_train, nadia_test, nadia_y_tr, nadia_y_te, nadia_h, nadia_w = get_face_nadia()
    mao_train, mao_test, mao_y_tr, mao_y_te, mao_h, mao_w = get_face_mao()
    lin_train, lin_test, lin_y_tr, lin_y_te, lin_h, lin_w = get_face_lin()
    kun_train, kun_test, kun_y_tr, kun_y_te,kun_h, kun_w = get_face_kun()
    ma_train, ma_test, ma_y_tr, ma_y_te, ma_h, ma_w = get_face_ma()
    train_x = [nadia_train, mao_train, lin_train, kun_train, ma_train]
    test_x = [nadia_test, mao_test, lin_test, kun_test, ma_test]
    train_y =[nadia_y_tr, mao_y_tr, lin_y_tr, kun_y_tr, ma_y_tr]
    test_y = [nadia_y_te, mao_y_te, lin_y_te, kun_y_te, ma_y_te]
    height = [nadia_h, mao_h, lin_h, kun_h, ma_h]
    width = [nadia_w, mao_w, lin_w, kun_w, ma_w]
    return train_x, test_x, train_y, test_y


"""new version"""
def load_faces_new():
    nadia_train, nadia_y_tr, nadia_h, nadia_w = get_face_nadia()
    mao_train, mao_y_tr, mao_h, mao_w = get_face_mao()
    lin_train, lin_y_tr, lin_h, lin_w = get_face_lin()
    kun_train, kun_y_tr,kun_h, kun_w = get_face_kun()
    ma_train, ma_y_tr, ma_h, ma_w = get_face_ma()
    yj_train, yj_y_tr, yj_h, yj_w = get_face_youngjun()
    ju_train, ju_y_tr, ju_h, ju_w = get_face_julier()
    gi_train, gi_y_tr, gi_h, gi_w = get_face_girl()
    ou_train, ou_y_tr, ou_h, ou_w = get_face_ou()
    ch_train, ch_y_tr, ch_h, ou_w = get_face_ch()
    train_x = [nadia_train, mao_train, lin_train, kun_train, ma_train, yj_train, ju_train, gi_train, ou_train, ch_train]
    # test_x = [nadia_test, mao_test, lin_test, kun_test, ma_test]
    train_y =[nadia_y_tr, mao_y_tr, lin_y_tr, kun_y_tr, ma_y_tr, yj_y_tr, ju_y_tr, gi_y_tr, ou_y_tr, ch_y_tr]
    # test_y = [nadia_y_te, mao_y_te, lin_y_te, kun_y_te, ma_y_te]
    height = [nadia_h, mao_h, lin_h, kun_h, ma_h]
    width = [nadia_w, mao_w, lin_w, kun_w, ma_w]
    return train_x, train_y


def preprocess():
    x_train, y_train = load_faces_new()

    """new version"""
    new_train_0 = np.zeros([130, 100, np.size(x_train[0], 2)])
    new_train_1 = np.zeros([130, 100, np.size(x_train[1], 2)])
    new_train_2 = np.zeros([130, 100, np.size(x_train[2], 2)])
    new_train_3 = np.zeros([130, 100, np.size(x_train[3], 2)])
    new_train_4 = np.zeros([130, 100, np.size(x_train[4], 2)])
    new_train_5 = np.zeros([130, 100, np.size(x_train[5], 2)])
    new_train_6 = np.zeros([130, 100, np.size(x_train[6], 2)])
    new_train_7 = np.zeros([130, 100, np.size(x_train[7], 2)])
    new_train_8 = np.zeros([130, 100, np.size(x_train[8], 2)])
    new_train_9 = np.zeros([130, 100, np.size(x_train[9], 2)])
    new_train = [new_train_0, new_train_1, new_train_2, new_train_3, new_train_4, new_train_5, new_train_6, new_train_7, new_train_8, new_train_9]
    for i in range(np.size(x_train)):
        for j in range(np.size(x_train[i], 2)):
            if i == 0 or i == 5 or i==6 or i==7 or i==9:
                new_train[i][:,:,j] = cv2.resize(x_train[i][:, :, j], (base_w, base_h), interpolation = cv2.INTER_CUBIC)
            else:
                new_train[i][:, :, j] = cv2.resize(x_train[i][:, :, j], (base_w, base_h), interpolation=cv2.INTER_AREA)

            # plt.imsave('./data/new_all_same_size_face_img/'+ str(int(y_train[i][j])) + '/' + str(i) + '_' + str(j),
            #            new_train[i][:, :, j])
            # save all new image
    # np.save('./data/x_train', new_train)
    # np.save('./data/y_train', y_train)
    print('done')
    return new_train, y_train


if __name__ == '__main__':
    preprocess()


"""old version"""
# new_train_0 = np.zeros((base_h, base_w, np.size(x_train[0], 2)))
# new_train_0 = new_train_0 - np.mean(new_train_0, 2)[:,:,None]
#
# new_train_1 = np.zeros((base_h, base_w, np.size(x_train[1], 2)))
# new_train_1 = new_train_1 - np.mean(new_train_1, 2)[:,:,None]
#
# new_train_2 = np.zeros((base_h, base_w, np.size(x_train[2], 2)))
# new_train_2 = new_train_2 - np.mean(new_train_2, 2)[:,:,None]
#
# new_train_3 = np.zeros((base_h, base_w, np.size(x_train[3], 2)))
# new_train_3 = new_train_0 - np.mean(new_train_3, 2)[:,:,None]
#
# new_train_4 = np.zeros((base_h, base_w, np.size(x_train[4], 2)))
# new_train_4 = new_train_0 - np.mean(new_train_4, 2)[:,:,None]
#
# new_train_0 += min([min(new_train_0), min(new_train_1), min(new_train_2), min(new_train_3), min(new_train_4)])
# new_train_1 += min([min(new_train_0), min(new_train_1), min(new_train_2), min(new_train_3), min(new_train_4)])
# new_train_2 += min([min(new_train_0), min(new_train_1), min(new_train_2), min(new_train_3), min(new_train_4)])
# new_train_3 += min([min(new_train_0), min(new_train_1), min(new_train_2), min(new_train_3), min(new_train_4)])
# new_train_4 += min([min(new_train_0), min(new_train_1), min(new_train_2), min(new_train_3), min(new_train_4)])


# x_train, x_test, y_train, y_test = load_faces()
# x_train = load_faces_new()
# x_train = load_faces()
# []不通过的原因是接收方的dim不同，函数没问题，起来新建一个接收方
# t_111= cv2.resize( new_train_0[:, :, 10] , (base_w, base_h), interpolation=cv2.INTER_CUBIC)
# plt.imsave('./data/new_same_size_face/111.png', t_111)
# new_train_0 = np.zeros((base_h, base_w, np.size(x_train[0], 2)))
# new_train_0 = new_train_0 - np.mean(new_train_0, 2)[:,:,None]
#
# new_train_1 = np.zeros((base_h, base_w, np.size(x_train[1], 2)))
# new_train_1 = new_train_1 - np.mean(new_train_1, 2)[:,:,None]
#
# new_train_2 = np.zeros((base_h, base_w, np.size(x_train[2], 2)))
# new_train_2 = new_train_2 - np.mean(new_train_2, 2)[:,:,None]
#
# new_train_3 = np.zeros((base_h, base_w, np.size(x_train[3], 2)))
# new_train_3 = new_train_0 - np.mean(new_train_3, 2)[:,:,None]
#
# new_train_4 = np.zeros((base_h, base_w, np.size(x_train[4], 2)))
# new_train_4 = new_train_0 - np.mean(new_train_4, 2)[:,:,None]
#
# new_train_0 += min([min(new_train_0), min(new_train_1), min(new_train_2), min(new_train_3), min(new_train_4)])
# new_train_1 += min([min(new_train_0), min(new_train_1), min(new_train_2), min(new_train_3), min(new_train_4)])
# new_train_2 += min([min(new_train_0), min(new_train_1), min(new_train_2), min(new_train_3), min(new_train_4)])
# new_train_3 += min([min(new_train_0), min(new_train_1), min(new_train_2), min(new_train_3), min(new_train_4)])
# new_train_4 += min([min(new_train_0), min(new_train_1), min(new_train_2), min(new_train_3), min(new_train_4)])
# for i in range(np.size(x_train)):
#     for j in range(np.size(x_train[i], 2)):
#         if i == 0:
#             new_train[i][:, :, j] = cv2.resize(new_train[i][:, :, j], (base_w, base_h), interpolation = cv2.INTER_CUBIC)
#         else:
#             new_train[i][:, :, j] = cv2.resize(new_train[i][:, :, j], (base_w, base_h), interpolation=cv2.INTER_AREA)
#         # save all new image
#         plt.imsave('./data/new_same_size_face/'+ str(int(y_train[i][j])) + '/' + str(i) + '_' + str(j) +'.png',
#                        new_train[i][:, :, j])