import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import plotly.plotly as py
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score
from numpy import arange
import seaborn as sns

sns.set(color_codes=True)

df = pd.read_csv("nursery.csv")

# Convert strings into frequency numbers
labelencoder=LabelEncoder()
for col in df.columns:
    df[col] = labelencoder.fit_transform(df[col])

# Split into train and test
train, test = train_test_split(df, test_size = 0.20, random_state=1)

# Train set
label = 'result'
train_y = train[label]
train_x = train[[x for x in train.columns if label not in x]]
# Test/Validation set
test_y = test[label]
test_x = test[[x for x in test.columns if label not in x]]

training_accuracy = []
validation_accuracy = []
test_accuracy = []
n_estimators = range(1,100)

# For N estimators in Boosting
for n in n_estimators:
    clf = AdaBoostClassifier(n_estimators=n, random_state=1)
        
    clf.fit(train_x, train_y)
    clf.predict(test_x)

    print n

    training_accuracy.append(accuracy_score(train_y, clf.predict(train_x)))
    cv = cross_val_score(clf, train_x, train_y, cv=7).mean()
    validation_accuracy.append(cv)
    test_accuracy.append(accuracy_score(test_y, clf.predict(test_x)))

temp_x = arange(3)
fig = plt.figure()
plt.style.use('ggplot')
line1, = plt.plot(n_estimators, training_accuracy, 'r', label="Training Accuracy")
line2, = plt.plot(n_estimators, validation_accuracy, 'b', label="Cross Validation Score")
line1, = plt.plot(n_estimators, test_accuracy, 'g', label="Testing Accuracy")
plt.xlabel('Number of Estimators')
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.title('Number of Estimators versus Accuracy (Nursery)')
fig.savefig('figures/nursery_bosting_estimator.png')
plt.close(fig)

# After finding the right estimator number, experiment on training set size
training_accuracy = []
validation_accuracy = []
test_accuracy = []
training_size = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

for s in training_size:
    # Define the classifier
    clf = AdaBoostClassifier(n_estimators=100, random_state=1)
    
    temp_train, _ = train_test_split(train, test_size= 1 - s, random_state=1)

    # Train set
    percent_train_y = temp_train[label]
    percent_train_x = temp_train[[x for x in train.columns if label not in x]]

    clf.fit(percent_train_x, percent_train_y)

    print 'Size: ', s, '%'

    training_accuracy.append(accuracy_score(percent_train_y, clf.predict(percent_train_x)))
    cv = cross_val_score(clf, percent_train_x, percent_train_y, cv=7).mean()
    validation_accuracy.append(cv)
    test_accuracy.append(accuracy_score(test_y, clf.predict(test_x)))

clf = AdaBoostClassifier(n_estimators=100, random_state=1)
clf.fit(train_x, train_y)

training_accuracy.append(accuracy_score(train_y, clf.predict(train_x)))
cv = cross_val_score(clf, train_x, train_y, cv=7).mean()
validation_accuracy.append(cv)
test_accuracy.append(accuracy_score(test_y, clf.predict(test_x)))
training_size.append(1)

fig = plt.figure()
line1, = plt.plot(training_size, training_accuracy, 'r', label="Training Accuracy")
line2, = plt.plot(training_size, validation_accuracy, 'b', label="Cross Validation Score")
line1, = plt.plot(training_size, test_accuracy, 'g', label="Testing Accuracy")
plt.xlabel('Training Set Size (%)')
plt.ylabel('Accuracy')
plt.title('Training size versus Accuracy (Nursery)')
plt.legend(loc='best')
fig.savefig('figures/nursery_bosting_trainingSize.png')
plt.close(fig)