import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
diabetes_dataset = pd.read_csv('diabetes.csv')
#1 mean that the patient is diabetic and 0 means that he is not diabetic
print(diabetes_dataset.head())
print(diabetes_dataset.shape)
#To describe the data
print(diabetes_dataset.describe())
print(diabetes_dataset['Outcome'].value_counts())
print(diabetes_dataset['Outcome'])
print(diabetes_dataset.groupby('Outcome').mean())

X=diabetes_dataset.drop('Outcome', axis = 1)
print(X)
Y = diabetes_dataset['Outcome']
print(Y)
# Now we need to pre-process the data
scaler = StandardScaler()
# We fit it to learn the parameters of the estimation
scaler.fit(X)
# Now we need to transform the data
data = scaler.transform(X)
X = data
print(X)
print(Y)
# Now we split it into train and test data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, stratify = Y, random_state = 2)
print(X.shape, X_train.shape, X_test.shape)
# Training the model
classifier = svm.SVC(kernel = 'linear')
#Training the SVM classifier
classifier.fit(X_train, Y_train)
#model evaluation
X_train_prediction= classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print("Accuracy score of the training data : ",training_data_accuracy)
X_test_prediction = classifier.predict(X_test)
testing_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print("Accuracy score for testing data :", testing_data_accuracy)
input_data = (10,168,74,0,0,38,0.537,34)
#change the input data to numpy array
input_data_as_n = np.asarray(input_data)
#Reshape the array
input_data_reshaped = input_data_as_n.reshape(1, -1)
std_data = scaler.transform(input_data_reshaped)
prediction = classifier.predict(std_data)
print(prediction)