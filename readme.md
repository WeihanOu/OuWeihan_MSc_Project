
This is the MSc program of Weihan Ou, 
Supervisors:	Nadia Bianchi-Berthouze
		Youngjun Cho
=========================================
Environment:
This progeam is writen in Python 3.5, IDE is Pycharm.
This program uses the following library to for runing:
	- Tensorflow
	- Numpy
	- openCV
	- Time
	- sklearn
	- random
=========================================
To run files:

To train and test models PCA_SVM, input "python ALL_PCA_SVM.py" in the command.
To train and test model VGGNet, input "python Th_CNN_no_PCA.py" in the command.
=========================================
Introduction of files:

1. Get_Face_[1, 2, 3, 4, 6, 7, 8, 9].py and Th_Get_Face_5.py
	These 10 files are used to load and cut thermal video frames, then extract facial areas.
	They call function "readImg()", "read_file()", "get_time_arr()", these functions are in file ReadImg.py.

2. Preprocessing.py 
	It is used to emsamble the above 10 files, and then resize the images into same size (130, 110) 
	It calls functions "get_face_[1, 2, 3, 4, 5, 6, 7, 8, 9]", these functions are the above 10 files.

3. ALL_PCA_SVM.py
	This file is used to form the training and testing dataset, and then do train and test on two models, PCA+SVM and SVM.
	This file calls function "preprocessing()" in the above file, to load in the facial images.

4. Th_CNN_no_PCA.py 
	It is used to form the training and testing dataset, and then construct, train, and test VGGNet.
	It calss function "preprocessing" in file "Preprocessing.py", to load in the facial images.