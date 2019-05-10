from sklearn.preprocessing import LabelEncoder
import glob
import os
import cv2 as cv
import mahotas
import numpy as np
from matplotlib import pyplot
import pickle

bins = 8

def fd_hu_moments(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    feature = cv.HuMoments(cv.moments(img)).flatten()
    return feature


def fd_haralick(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    feature = mahotas.features.haralick(img).mean(axis=0)
    return feature


def fd_histogram(img,mask = None):
    img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    hist = cv.calcHist([img], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    cv.normalize(hist, hist)
    return hist.flatten()

test_path = "17flowers/Predict"

all_labels = os.listdir("17flowers/train")
le = LabelEncoder()
transformed_labels = le.fit_transform(all_labels)

infile = open('model', 'rb')
clf = pickle.load(infile)
infile.close()


for f in glob.glob(test_path+"/*.jpg"):
    image = cv.imread(f)
    image = cv.resize(image, (500,500))

    f1 = fd_hu_moments(image)
    f2 = fd_haralick(image)
    f3 = fd_histogram(image)

    global_feature = np.hstack([f1, f2, f3])
    prediction = clf.predict(global_feature.reshape(1, -1))[0]

    cv.putText(image, all_labels[np.where(transformed_labels == prediction)[0][0]], (20, 20), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
    pyplot.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    pyplot.show()