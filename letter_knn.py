import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn import neighbors
from sklearn.model_selection import cross_val_score
import seaborn as sns

sns.set(color_codes=True)

df = pd.read_csv("letter.csv")

# Convert strings into frequency numbers
labelencoder=LabelEncoder()
df['lettr'] = labelencoder.fit_transform(df['lettr'])

# Split into train and test
train, test = train_test_split(df, test_size = 0.20, random_state=1)

label = 'lettr'
# Train set
train_y = train[label]
train_x = train[[x for x in train.columns if label not in x]]
# Test/Validation set
test_y = test[label]
test_x = test[[x for x in test.columns if label not in x]]

training_accuracy = []
validation_accuracy = []
test_accuracy = []
k_values = range(1,35,2)

# # For knn, experiment on different k values
# for k in k_values:

#     # Define the classifier
#     clf = neighbors.KNeighborsClassifier(k)
#     clf.fit(train_x, train_y)

#     print 'K: ', k

#     training_accuracy.append(accuracy_score(train_y, clf.predict(train_x)))
#     cv = cross_val_score(clf, train_x, train_y, cv=7).mean()
#     validation_accuracy.append(cv)
#     test_accuracy.append(accuracy_score(test_y, clf.predict(test_x)))

# # Plot the k values graph
# plt.style.use('ggplot')
# fig = plt.figure()
# line1, = plt.plot(k_values, training_accuracy, 'r', label="Training Accuracy")
# line2, = plt.plot(k_values, validation_accuracy, 'b', label="Cross Validation Score")
# line1, = plt.plot(k_values, test_accuracy, 'g', label="Testing Accuracy")
# plt.xlabel('K-Nearest')
# plt.ylabel('Accuracy')
# plt.title('Number of K\'s versus Accuracy (Letter)')
# plt.legend(loc='best')
# fig.savefig('figures/letter_knn_knumber.png')
# plt.close(fig)


# After finding the right k value, experiment on training set size
k = 2
training_accuracy = []
validation_accuracy = []
test_accuracy = []
training_size = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

print "--- KNN ---"
for s in training_size:
    # Define the classifier
    clf = neighbors.KNeighborsClassifier(k)
    
    temp_train, _ = train_test_split(train, test_size= 1 - s, random_state=1)

    # Train set
    percent_train_y = temp_train[label]
    percent_train_x = temp_train[[x for x in train.columns if label not in x]]

    print percent_train_x.shape

    clf.fit(percent_train_x, percent_train_y)

    print 'Size: ', s, '%'
    print accuracy_score(test_y, clf.predict(test_x))

    training_accuracy.append(accuracy_score(percent_train_y, clf.predict(percent_train_x)))
    cv = cross_val_score(clf, percent_train_x, percent_train_y, cv=7).mean()
    validation_accuracy.append(cv)
    test_accuracy.append(accuracy_score(test_y, clf.predict(test_x)))

clf = neighbors.KNeighborsClassifier(k)
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
plt.title('Training size versus Accuracy (Letter)')
plt.legend(loc='best')
fig.savefig('figures/letter_knn_trainingSize.png')
plt.close(fig)