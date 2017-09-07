import glob
import numpy as np
from matplotlib import pyplot as plt
# from PIL import Image
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


# def exam_read_file(f_0, f_1, f_2):
#     # read train data from outside file
#     list_0 = glob.glob(file_0)
#     # imgs_0 = np.array([np.array(Image.open(fname).convert('L')) for fname in list_0])
#     imgs_0 = np.array([np.array(np.load(fname)) for fname in list_0])
#     imgs_0 = np.transpose(imgs_0, [1, 2, 0])
#     list_1 = glob.glob(file_1)
#     imgs_1 = np.array([np.array(np.load(fname)) for fname in list_1])
#     imgs_1 = np.transpose(imgs_1, [1, 2, 0])
#     list_2 = glob.glob(file_2)
#     imgs_2 = np.array([np.array(np.load(fname)) for fname in list_2])
#     imgs_2 = np.transpose(imgs_2, [1, 2, 0])
#     return imgs_0, imgs_1, imgs_2


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
    idx_7_1 = np.reshape(random.sample(range(787, 2090), n_img), [-1, 1])
    idx_7_2 = np.reshape(random.sample(range(2090, np.size(x[4], 2)), n_img), [-1, 1])

    idx_8_0 = np.reshape(random.sample(range(0, 885), n_img), [-1, 1])
    idx_8_1 = np.reshape(random.sample(range(885, 2377), n_img), [-1, 1])
    idx_8_2 = np.reshape(random.sample(range(2377, np.size(x[4], 2)), n_img), [-1, 1])

    idx_9_0 = np.reshape(random.sample(range(0, 807), n_img), [-1, 1])
    idx_9_1 = np.reshape(random.sample(range(785, 2775), n_img), [-1, 1])
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
    for this_ex in range(10):
        # tmp = x[this_ex] - np.mean(x[this_ex], 2)[:,:,None]
        x[this_ex] -= np.mean(x[this_ex], 2)[:,:,None]

    for this_choose in range(5):
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
                 np.concatenate((x_9_0, x_9_1, x_9_2), axis=2)
                 ]
        new_y = np.zeros(int(3*9*n_img))
        for i in range(3*9):
            new_y[int(i*n_img):int((i+1)*n_img)] = int(i % 3)

        for fold_j in range(0, 10):
            x_te = new_x[fold_j]
            y_te = new_y[0:n_img*3]
            idx = list(set(np.arange(10))-set(np.arange(fold_j, fold_j+1)))
            for k in range(9):
                this_x_tr = new_x[idx[k]]
                np.save('./data/new_all_same_size_face/0/'+str(k)+'xtr', this_x_tr)
                this_y_tr = new_y[idx[k]]
                np.save('./data/new_all_same_size_face/0/'+str(k)+'ytr', this_y_tr)

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

            start = time.time()
            ei_face, x_tr_pca, x_te_pca, var_per = pca(x_tr, x_te, np.size(x_tr, 0), np.size(x_tr, 1))
            pca_stop = time.time()
            print('var_per, '+str(fold_j)+'time: ', var_per)
            print('sum_var_per, '+str(fold_j)+'time: ', sum(var_per))
            for i_save in range(np.size(ei_face, 0)):
                this_eg = ei_face[i_save, :]
                plt.imsave('./data/eigen/new' + str(this_choose) + str(fold_j) +"_"+str(i_save) + '.png', (np.reshape(this_eg, [130, 100])))
                a=0
            classifier_pca = LinearSVC()
            classifier_pca.fit(x_tr_pca, new_y)
            train_accuracy = classifier_pca.score(x_tr_pca, new_y)
            test_accuracy = classifier_pca.score(x_te_pca, y_te)
            stop = time.time()
            predict_pca = classifier_pca.predict(x_te_pca)
            con_mat = confusion_matrix(y_te, predict_pca)
            print("pca time, " +str(fold_j)+'time: ', str(int(pca_stop - start)))
            print("with p total time: "+str(fold_j)+'time: ', str(int(stop-start)))
            print("pca_train_acc: "+str(fold_j)+'time: ', train_accuracy)
            print("pca_test_acc"+str(fold_j)+'time: ', test_accuracy)
            print("conf mat_pca: ", con_mat)

            start_ = time.time()
            x_tr_nop = np.transpose(np.reshape(x_tr, [130 * 100, -1]))
            x_te_nop = np.transpose(np.reshape(x_te, [130 * 100, -1]))
            classifier = LinearSVC()
            classifier.fit(x_tr_nop, new_y)
            train_accuracy = classifier.score(x_tr_nop, new_y)
            test_accuracy = classifier.score(x_te_nop, y_te)
            stop_ = time.time()
            predict = classifier.predict(x_te_nop)
            con_mat_nop = confusion_matrix(y_te, predict)
            print("nop time: "+str(fold_j)+'time: ', str(int(stop_ - start_)))
            print("nop_train_acc"+str(fold_j)+'time: ', train_accuracy)
            print("nop_train_acc"+str(fold_j)+'time: ', test_accuracy)
            print("conf mat_no_pca: ", con_mat_nop)
            print(" ")
            print("==================================================")
            print(" ")


if __name__ == '__main__':
    main()