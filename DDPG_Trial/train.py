import time
import pandas as py 
import ddpg_tf
from ddpg_tf import Agent
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
import matplotlib.pyplot as plt
import pandas as pd
from data import Import_Data
from utils import plotLearning, plotRPMandPrediction

#Intialize DDPG Agent
agent = Agent(alpha=0.000001, beta=0.00001, input_dims=[3], tau=0.005,
              batch_size=120, layer1_size=800, layer2_size=600,
              n_actions=1)

#Importing the DataSet
csv_file_path_1 = "C:/Users/keegh/Documents/Orbtronics_Agri_Sensor/DRL_Test_Algorithms/training data/Random Step/random_ramp_test_17min.CSV" # Replace with the actual path to your Excel file
df_1 = pd.read_csv(csv_file_path_1, skiprows=13, header=0, usecols=[3,4,6,7], nrows=60000) # Load the Excel sheet, excluding the specified column


#Combine DataSet 
features = pd.concat([df_1], axis=0)

# Assume X_train, y_train, X_test, y_test are your data
X_train_o, X_test_o, y_train, y_test = train_test_split(features[['Diameter','Setpoint','Preform_Idler_Speed']],features[['Measured_rpm_Filtered']], test_size=0.1, random_state=42, shuffle=False)

#Normalize the Data
scaler = preprocessing.StandardScaler().fit(X_train_o)
X_train = scaler.transform(X_train_o)
scaler_2 = preprocessing.StandardScaler().fit(X_test_o)
X_test = scaler_2.transform(X_test_o)

#Load Previously saved models
#agent.load_models()

start_time = time.time()

#Training Loop
num_epochs = 2
X_train = pd.DataFrame(X_train_o)
y_train = pd.DataFrame(y_train)
X_train = X_train.dropna()
y_train = y_train.dropna()

score_history = []
counter_history = []
rpms = []
predictions = []
correctPredictions = 0 
totalPredictions=0

for epoch in range(num_epochs):
    score = 0
    counter = 0
    for state, act in zip(X_train.values[:-1], y_train.values[:-1]):
        #Change State list into a Numpy Array
        state = np.array(state)
        state = np.array(state, dtype=np.float32)
               
        #Select action
        action = agent.choose_action(state)
        action = np.mean(y_train.values)*action*2 + np.std(y_train.values)*2
        #d = np.array(y_train.max())-np.array(y_train.min())
        #d = 250 - 50 Hard coded max and minimum range of the spooling motor
        #action = ((action+0.4)/0.75)*d + np.array(y_train.min())

        if action < 15:
            action = np.array([action[0]+100])
        elif action > 300:
            action = np.array([action[0]-100])

        #Observe new state
        if counter+1< X_train.shape[0]:
            new_state = X_train.iloc[counter+1].values
        else:
            new_state = np.array([X_train.mean[0],X_train.mean[1]])

        #Calculate Error 
        if counter + 1 < len(y_train):
            error = int(y_train.iloc[counter+1]) - int(action) #rpm - predicted rpm
        else:
            new_state= np.array([X_train.mean[0],X_train.mean[1]])

        #Calculate Accuracy 
        if abs(error)>10:
            totalPredictions +=1
        else:
            correctPredictions += 1
            totalPredictions+= 1

        #Calculate Reward
        reward = -abs(error)

        #Store the experience
        agent.remember(state, action, reward, new_state)

        #Learn from the experience
        agent.learn()

        score = -reward

        if counter +1 < len(y_train):
            rpms.append(y_train.iloc[counter+1])
            predictions.append(action)

        print("Count|", counter, "|epoch|", epoch, "/", num_epochs, "|action|", action, "action_raw", agent.choose_action(state), "|RPM at t+1|",int(y_train.iloc[counter+1]), "|error", error)
        counter += 1

    score_history.append(score)
    print('epoch ', epoch, 'Score %.2f' % score,
              'trailing 5 data points avg %.3f' % np.mean(score_history[-100:]))
    accuracy = (correctPredictions/totalPredictions)*100
    print('Accuracy Score: %.2f' % accuracy, '%')

filename = 'DDPG Training_Avg Error vs Time .png'
plotLearning(score_history, filename, window=100)
plotRPMandPrediction(rpms, predictions, filename="RPM and Predicted RPMs", window=100)

end_time = time.time()
duration = (end_time - start_time)/60
print("Time taken to train the model", duration, "mins")

#Save the model
agent.save_models()

