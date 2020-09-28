import numpy as np
import pandas as pd
import tensorflow as tf
tf.__version__

import sys
print(sys.path)

# Importing the dataset
dataset = pd.read_csv('https://raw.githubusercontent.com/thaiseq/ChurnAnalysis/master/churn.csv')
X = dataset.iloc[0:4500, 4:-1].values
y = dataset.iloc[0:4500, -1].values
testy = dataset.iloc[4500:5000, -1].values
print(X)
print(y)
print(dataset.iloc[0, 4:-1].values)


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Building the ANN

# Initializing the ANN
ann = tf.keras.models.Sequential()

# Adding the input layer and the first hidden layer
ann.add(tf.keras.layers.Dense(units=12, activation='relu'))

# Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=12, activation='relu'))

ann.add(tf.keras.layers.Dense(units=12, activation='relu'))

# Adding the output layer
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Part 3 - Training the ANN

# Compiling the ANN
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Training the ANN on the Training set
ann.fit(X_train, y_train, batch_size = 32, epochs = 200)
custarray = []
for x in range(4500, 5000):
    arr = dataset.iloc[x, 4:-1].values
    custarray.append(arr)
print('  GUESS    REAL VALUE     COMPARISON          ')
j = 0
predictedexits = 0
missedexits = 0
for x in custarray:
    denoter = ''
    prediction = float((ann.predict(sc.transform([x]))))
    roundedpre = round(prediction)
    if roundedpre == testy[j]:
        denoter = '   '+str(testy[j])+'    MATCH'
        if testy[j] == 1:
            predictedexits = predictedexits + 1
    else:
        denoter = '   '+str(testy[j])+'    DOES NOT MATCH'
        if testy[j] == 1:
            missedexits = missedexits + 1
    print('Customer '+str(j+1)+': '+str(prediction)+denoter)
    j=j+1

print('Successfully predicted '+str(predictedexits)+' out of '+str(predictedexits+missedexits)+' discontinuations in cohort. Accuracy: ' +str((predictedexits/(predictedexits+missedexits))*100)+'%')


# Predicting the Test set results
y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)

