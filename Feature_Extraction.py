from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import numpy as np
import mahotas
import cv2 as cv
import os
import h5py

image_size = tuple((500, 500))
path = "17flowers/train"
n_trees = 100
bins = 8
test_size = .10
seed = 9


def fd_hu_moments(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    feature = cv.HuMoments(cv.moments(img)).flatten()
    return feature


def fd_haralick(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    feature = mahotas.features.haralick(img).mean(axis=0)
    return feature


def fd_histogram(img, mask=None ):
    img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    hist = cv.calcHist([img], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    cv.normalize(hist, hist)
    return hist.flatten()


train_labels = os.listdir(path)
train_labels.sort()
print(train_labels)

global_features = []
labels = []

i = 0
j = 0
k = 0

class_size = 80

for p in train_labels:
    full_folder_path = os.path.join(path, p)
    current_label = p
    k = 1

    for x in os.listdir(full_folder_path):
        file = full_folder_path + "/" + str(x)
        image = cv.imread(file)
        image = cv.resize(image, image_size)

        f1 = fd_hu_moments(image)
        f2 = fd_haralick(image)
        f3 = fd_histogram(image)

        global_feature = np.hstack([f1, f2, f3])
        labels.append(j)
        global_features.append(global_feature)

        i += 1
        k += 1
    print("Processed folder : " + p)
    j += 1

le = LabelEncoder()
target = le.fit_transform(labels)
print(target)
print(target.shape)
scaler = MinMaxScaler(feature_range=(0, 1))
rescaled_features = scaler.fit_transform(global_features)

h5f_data = h5py.File('output/data.h5', 'w')
h5f_data.create_dataset('dataset_1', data=np.array(rescaled_features))

h5f_label = h5py.File('output/labels.h5', 'w')
h5f_label.create_dataset('dataset_1', data=np.array(target))

h5f_data.close()
h5f_label.close()

print('Training is Completed')