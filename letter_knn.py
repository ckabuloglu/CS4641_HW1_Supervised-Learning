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

training_accuracy = []
validation_accuracy = []
test_accuracy = []
k_values = range(1,41,2)

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

plt.style.use('ggplot')
fig = plt.figure()
line1, = plt.plot(k_values, training_accuracy, 'r', label="Training Accuracy")
line2, = plt.plot(k_values, validation_accuracy, 'b', label="Validation Accuracy")
line1, = plt.plot(k_values, test_accuracy, 'g', label="Testing Accuracy")
plt.xlabel('K-Nearest')
plt.ylabel('Accuracy')
plt.title('Number of K\'s versus Accuracy')
plt.legend(bbox_to_anchor=(0.8, 0.6), bbox_transform=plt.gcf().transFigure)
fig.savefig('figures/letter_knn_knumber.png')
plt.close(fig)



