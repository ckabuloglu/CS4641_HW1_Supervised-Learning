import csv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn import neighbors
from sklearn import svm

df = pd.read_csv("letter.csv")

# Convert strings into frequency numbers
labelencoder=LabelEncoder()
# for col in df.columns:
df['lettr'] = labelencoder.fit_transform(df['lettr'])

# Split into train and test
train, test = train_test_split(df, test_size = 0.20)

# Train set
label = 'lettr'
train_y = train[label]
train_x = train[[x for x in train.columns if label not in x]]
# Test/Validation set
test_y = test[label]
test_x = test[[x for x in test.columns if label not in x]]

print test_x.shape
print test_y.shape
print train_x.shape
print train_y.shape

# For knn
print "--- KNN ---"
for n_neighbors in range(1,21,2):
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights='distance')
    clf.fit(train_x, train_y)

    print 'K:', accuracy_score(test_y, clf.predict(test_x))

# For svm
print '--- SVM ---'
for kernel in ['linear','poly','rbf']:
    if kernel == 'poly':
        for d in range(1,4):
            clf = svm.SVC(kernel='poly', degree=d)
            clf.fit(train_x, train_y)
            clf.predict(test_x)
            print kernel, ', deg:', str(d),'-', accuracy_score(test_y, clf.predict(test_x))
    else:
        clf = svm.SVC(kernel=kernel)
        clf.fit(train_x, train_y)
        clf.predict(test_x)
        print kernel,'-', accuracy_score(test_y, clf.predict(test_x))



