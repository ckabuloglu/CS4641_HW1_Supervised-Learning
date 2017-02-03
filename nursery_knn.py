import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn import neighbors
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
import seaborn as sns

sns.set(color_codes=True)

df = pd.read_csv("nursery.csv")

# Convert strings into frequency numbers
labelencoder=LabelEncoder()
for col in df.columns:
    df[col] = labelencoder.fit_transform(df[col])

print df.shape

# Split into train and test (75% train - 6499 rows, 25% test - 1625 rows)
train, test = train_test_split(df, test_size = 0.25)

label = 'result'
# General x,y
data_y = df[label]
data_x = df[[x for x in train.columns if label not in x]]
# Train set
train_y = train[label]
train_x = train[[x for x in train.columns if label not in x]]
# Test/Validation set
test_y = test[label]
test_x = test[[x for x in test.columns if label not in x]]

training_accuracy = []
validation_accuracy = []
test_accuracy = []
k_values = range(1,11,2)

# For knn
print "--- KNN ---"
for k in k_values:
    # Define the classifier
    clf = neighbors.KNeighborsClassifier(k, weights='distance')
    clf.fit(train_x, train_y)

    print 'K: ', k

    training_accuracy.append(accuracy_score(train_y, clf.predict(train_x)))
    cv = cross_val_score(clf, train_x, train_y, cv=7).mean()
    validation_accuracy.append(cv)
    test_accuracy.append(accuracy_score(test_y, clf.predict(test_x)))

line1, = plt.plot(k_values, training_accuracy, 'r', label="Training Accuracy")
line2, = plt.plot(k_values, validation_accuracy, 'b', label="Validation Accuracy")
line1, = plt.plot(k_values, test_accuracy, 'g', label="Testing Accuracy")
plt.legend(bbox_to_anchor=(0.92, 0.25), bbox_transform=plt.gcf().transFigure)
plt.show()

