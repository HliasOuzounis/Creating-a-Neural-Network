import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

from My_Neural_Network.My_Neural_Network import Neural_Network

data = pd.read_csv("Fish_dataset.csv") #loading the dataset from the csv file

y = data["Weight"].to_numpy() #saving the target value, the weight of each fish

#Because there are multiple species of fish in the dataset and their names are used
#We replace the name with a value the neural network can use like the average weight of each fish species
avg_weights = [0, 0, 0, 0, 0, 0, 0]
nof_fish = [0, 0, 0, 0, 0, 0, 0]

for i, fish in enumerate(data["Species"]):
    if fish == "Bream":
        avg_weights[0] += y[i]
        nof_fish[0] += 1
    elif fish == "Roach":
        avg_weights[1] += y[i]
        nof_fish[1] += 1
    elif fish == "Whitefish":
        avg_weights[2] += y[i]
        nof_fish[2] += 1
    elif fish == "Parkki":
        avg_weights[3] += y[i]
        nof_fish[3] += 1
    elif fish == "Perch":
        avg_weights[4] += y[i]
        nof_fish[4] += 1
    elif fish == "Pike":
        avg_weights[5] += y[i]
        nof_fish[5] += 1
    elif fish == "Smelt":
        avg_weights[6] += y[i]
        nof_fish[6] += 1

avg_weights = [i/j for i, j in zip(avg_weights, nof_fish)] #calculating the average weight of each fish type

#changing the value of the species to the corresponding average weight
data.loc[data["Species"] == "Bream", "Species"] = avg_weights[0]
data.loc[data["Species"] == "Roach", "Species"] = avg_weights[1]
data.loc[data["Species"] == "Whitefish", "Species"] = avg_weights[2]
data.loc[data["Species"] == "Parkki", "Species"] = avg_weights[3]
data.loc[data["Species"] == "Perch", "Species"] = avg_weights[4]
data.loc[data["Species"] == "Pike", "Species"] = avg_weights[5]
data.loc[data["Species"] == "Smelt", "Species"] = avg_weights[6] 

X = data.loc[:, data.columns != "Weight"].to_numpy()

y = y.reshape(-1, 1)#reshaping the array to 2d

scalerX = StandardScaler().fit(X)
scalery = StandardScaler().fit(y)
X_st = scalerX.transform(X)
y_st = scalery.transform(y) #scaling the data for better results

X_train, X_test, y_train, y_test = train_test_split(X_st, y_st, test_size = 0.20, random_state = 42) #training - 80% testing - 20%

nn = Neural_Network((15, 15), function = "relu", learning_rate = 0.1, random_state = 1, nof_iterations = 200)#creating the neural network class
nn.fit(X_train, y_train) #training the neural network

fig, axs = plt.subplots(1, 2, figsize = (12, 5)) #plotting the results

axs[0].set_title("Accuracy My Neural Network, score: {0:.2f}".format(nn.score(X_test, y_test)), fontsize = 17)
points = (scalery.inverse_transform(nn.predict(X_test)), scalery.inverse_transform(y_test))
#(the values predicted by the model, the real values)
axs[0].axis([-100.0, 1410.0, -100.0, 1410.0])
axs[0].scatter(points[0], points[1], s = 42) #plot the points

axs[0].plot([-80, 1350], [-80, 1350], "r")
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