import h5py
import numpy as np
from matplotlib import pyplot
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
import pickle


models = []
models.append(('SVM', SVC(random_state=9)))
models.append(('KNN', KNeighborsClassifier()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('LR', LogisticRegression(random_state=9)))
models.append(('RF', RandomForestClassifier(n_estimators=100, random_state=42)))


results = []
names = []
scoring = 'accuracy'

h5f_data = h5py.File('output/data.h5', 'r')
h5f_label = h5py.File('output/labels.h5', 'r')

global_features_string = h5f_data['dataset_1']
global_labels_string = h5f_label['dataset_1']

global_features = np.array(global_features_string)
global_labels = np.array(global_labels_string)


h5f_data.close()
h5f_label.close()

(trainDataGlobal, testDataGlobal, trainLabelGlobal, testLabelGlobal) = train_test_split(np.array(global_features),np.array(global_labels),test_size=0.3, random_state=42)

import warnings
warnings.filterwarnings('ignore')

print('Model    Mean_Accuracy     Standard_Deviation')
for name, model in models:
    kfold = KFold(n_splits=10, random_state=7)
    cv_results = cross_val_score(model, trainDataGlobal, trainLabelGlobal, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print(name, cv_results.mean(), cv_results.std())

fig = pyplot.figure()
fig.suptitle('ML algorithm comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()

########################################################################################################################
#We are using RandomForestClassifier because it has more accuracy compared to other classifiers for the given 17flowers dataset
clf = RandomForestClassifier(n_estimators=100, random_state=42)

clf.fit(trainDataGlobal, trainLabelGlobal)

outfile = open('model', 'wb')
pickle.dump(clf, outfile)
outfile.close()
