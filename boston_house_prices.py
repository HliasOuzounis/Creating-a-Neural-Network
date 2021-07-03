import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from My_Neural_Network.My_Neural_Network import Neural_Network

X, y = load_boston(return_X_y=True) #loading the dataset

y = y.reshape(-1, 1) #reshaping the array to scale it

scalerX = StandardScaler().fit(X)
scalery = StandardScaler().fit(y)
X_st = scalerX.transform(X)
y_st = scalery.transform(y) #scaling the arrays for better results

X_train, X_test, y_train, y_test = train_test_split(X_st, y_st, test_size = 0.20, random_state = 1) #training - 80% testing - 20%

nn = Neural_Network((15, 15), function = "relu", learning_rate = 0.1, random_state = 3)  #creating the neural network class
nn.fit(X_train, y_train) #training the neural network
 
fig, axs = plt.subplots(1, 2, figsize = (12, 5)) #plotting the results

axs[0].set_title("Accuracy My Neural Network, score: {0:.2f}".format(nn.score(X_test, y_test)), fontsize = 17)
points = (scalery.inverse_transform(nn.predict(X_test)), scalery.inverse_transform(y_test))
#(the values predicted by the model, the real values)
axs[0].axis([0.0, 55.0, .0, 55.0])
axs[0].scatter(points[0], points[1], s = 42) #plots the points

axs[0].plot([2, 53], [2, 53], "r") 
#plots y = x as a red line
#the closer the point is to this line, the better the prediction
#If it's on the line: predicted value = real value

axs[0].set_xlabel("predictions", fontsize = 12)
axs[0].set_ylabel("real values", fontsize = 12)
axs[0].grid()

axs[1].set_title("Loss curve for the Neural Network", fontsize = 17)
axs[1].plot([i for i in range (nn.iterations)], nn.loss_curve, "r") 
axs[1].set_xlabel("iterations", fontsize = 12)
axs[1].set_ylabel("loss", fontsize = 12)
#plots the loss curve of the neural network for each iteration. 
# From this curve we can see how fast and accurately the network learns

plt.show()
