import numpy as np
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from ReadImg import *


def pca(x_train, x_test, frame_height, frame_width):
    # pca_input: 2184*(240*320)
    x_train = x_train.reshape((frame_height*frame_width, np.size(x_train, 2)))
    x_train = np.transpose(x_train)
    x_test = x_test.reshape((frame_height*frame_width, np.size(x_test, 2)))
    x_test = np.transpose(x_test)
    # calculating eigen faces
    obj_pca = PCA(n_components=0.95)
    # pca projection
    _ = obj_pca.fit(x_train)
    x_train_pca = obj_pca.transform(x_train)
    x_test_pca = obj_pca.transform(x_test)
    # eigen faces
    eigen_faces = obj_pca.components_
    var_percent = obj_pca.explained_variance_ratio_
    return eigen_faces, x_train_pca, x_test_pca, var_percent


# def extract_region(n_au_height, n_au_width):
#     this_region = np.zeros((n_au_height, n_au_width))
#     # extract certain regions that are related to the corresponding AUs
#     # need pre-processing to align faces in each image into same position first
#     return this_region


