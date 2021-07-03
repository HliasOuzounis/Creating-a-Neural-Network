from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from My_Neural_Network.My_Neural_Network import Neural_Network

X, y = load_iris(return_X_y=True) #loading the dataset
#0 = Iris Setos
#1 = Iris versicolour
#2 = Iris Virginica

scaler = StandardScaler().fit(X)
X_st = scaler.transform(X) #scaling the data for better results

X_train, X_test, y_train, y_test = train_test_split(X_st, y, test_size = 0.20) #training - 80% testing - 20%

nn = Neural_Network(nof_iterations = 200) #creating the neural network class

nn.fit(X_train, y_train) #training the neural network

predictions = [round(i) for i in nn.predict(X_test)[:, 0]] 
#The model returns any value so we round them to the nearest integer and that is the models prediction

print("Real values:\n", y_test) #printing the real values
print("Predictions:\n", predictions) #and those predicted by the neural netowrk

print(y_test == predictions) #printing a true-false table for accuracy

accuracy = sum([i == j for i, j in zip(predictions, y_test)])/len(y_test) #calculates the percent of correct guesses
print("Accuracy: {0:.2f}".format(accuracy)) 

