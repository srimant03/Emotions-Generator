from sklearn.feature_extraction.text import TfidfVectorizer

with open('/content/drive/MyDrive/NLPA1/corpus.txt','r') as f:
  corpus = f.readlines()

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)

print(X.shape)

emotion_mapping = {'sadness': -1, 'joy': 1, 'anger': -2, 'love': 2, 'fear': -3, 'surprise': 3}

import numpy as np
y = np.zeros(2400)

with open('/content/drive/MyDrive/NLPA1/labels.txt','r') as f:
  labels = f.readlines()

for i in range(len(labels)):
  y[i] = emotion_mapping[labels[i].strip()]

indices = np.random.permutation(len(y))

shuffled_X= X[indices]
shuffled_y = y[indices]

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an SVC model
svc = SVC(kernel='linear', C=1.0, random_state=42)

# Train the model on the training data
svc.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = svc.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

emotion_list = ['sadness', 'joy', 'anger', 'love', 'fear','surprise']
corpus =[]
for i in emotion_list:
  filename = '/content/drive/MyDrive/NLPA1/gen_'+i+'.txt'
  with open(filename, 'r') as f:
    corpus.extend(f.readlines())

X_test = vectorizer.transform(corpus)

print(X_test.shape)

y_test = np.zeros(300)

cnt = 0
for i in emotion_list:
  for j in range(50):
    y_test[j+cnt*50] = emotion_mapping[i]
  cnt+=1

X_train = shuffled_X
y_train = shuffled_y

svc = SVC(kernel='linear', C=1.0, random_state=42)
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

f1_macro = f1_score(y_test, y_pred, average='macro')
print("Macro F1 Score:", f1_macro)

precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average=None)
for class_label in range(len(precision)):
    print(f"Class {class_label} - Precision: {precision[class_label]}, Recall: {recall[class_label]}, F1 Score: {f1[class_label]}, Support: {support[class_label]}")

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

param_grid = {'C': [0.1, 1, 10, 100, 1000],
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['poly','linear','rbf']}

grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3)

grid.fit(X_train, y_train)

grid_predictions = grid.predict(X_test)

print(classification_report(y_test, grid_predictions))

print(grid.best_params_)