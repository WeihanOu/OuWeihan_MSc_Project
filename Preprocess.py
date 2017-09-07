from Get_Face_1 import *
from Get_Face_2 import *
from Get_Face_3 import *
from Get_Face_4 import *
from Th_Get_Face_5 import *
from Get_Face_6 import *
from Get_Face_7 import *
from Get_Face_8 import *
from Get_Face_9 import *
from Get_Face_10 import *
from PIL import Image
import cv2
base_w = 100
base_h = 130



"""new version"""
def load_faces_new():
    train_7, y_tr_7, h_7, w_7 = get_face_7()
    train_6, y_tr_6, h_6, w_6 = get_face_6()
    train_4, y_tr_4, h_4, w_4 = get_face_4()
    train_3, y_tr_3,h_3, w_3 = get_face_3()
    train_5, y_tr_5, h_5, w_5 = get_face_5()
    train_9, y_tr_9, h_9, w_9 = get_face_9()
    train_10, y_tr_10, h_10, w_10 = get_face_10()
    train_2, y_tr_2, h_2, w_2 = get_face_2()
    train_8, y_tr_8, h_8, w_8 = get_face_8()
    train_1, y_tr_1, h_1, w_1 = get_face_1()
    train_x = [train_7, train_6, train_4, train_3, train_5, train_9, train_10, train_2, train_8, train_1]
    train_y =[y_tr_7, y_tr_6, y_tr_4, y_tr_3, y_tr_5, y_tr_9, y_tr_10, y_tr_2, y_tr_8, y_tr_1]
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