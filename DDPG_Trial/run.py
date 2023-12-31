import sys
import ddpg_tf
from ddpg_tf import Agent
sys.path.insert(0, "..")
from opcua import Client
import time
import tensorflow as tf
import numpy as np
import time
import pandas as pd
import csv
from datetime import datetime

if __name__ == "__main__":
    #Connect to the client
    client = Client("opc.tcp://DESKTOP-A9QNR1L:4990/FactoryTalkLinxGateway1")
    #Intialize DDPG Agent
    agent = Agent(alpha=0.000001, beta=0.00001, input_dims=[3], tau=0.005,
                batch_size=500, layer1_size=800, layer2_size=600,
                n_actions=1)
    # Create a CSV file for saving the data
    csv_file = open('Model 27_Test2_Learning_Batch Size 500_2.csv', 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Time','RPM from Model','RPM from PLC','Setpoint','Diameter','Setpoint - Diameter','Preform Speed','Actual Speed'])
        
    try:
        client.connect()
        root = client.get_root_node()
        setpoint = client.get_node("ns=2;s=[opc_server]setpoint_diameter")
        diameter = client.get_node("ns=2;s=[opc_server]Diameter")
        preform_idler_speed = client.get_node("ns=2;s=[opc_server]Preform_Idler_Speed") #input address for preform idler 
        spool_speed_Model = client.get_node("ns=2;s=[opc_server]DC_output")
        spool_speed_PLC = client.get_node("ns=2;s=[opc_server]PLC_output")
        actual_speed = client.get_node("ns=2;s=[opc_server]Measured_rpm_Filtered")
        agent.load_models() 
        score_history = []
        counter_history = []
        rpms = []
        predictions = []
        correctPredictions = 0 
        totalPredictions=0

        while True:
            try:
                start_time = datetime.now()
                s = start_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]# Trimming to milliseconds
                start_time_1 = time.time()
                state = [setpoint.get_value(),diameter.get_value(), preform_idler_speed.get_value()]
                rpm_from_plc = spool_speed_PLC.get_value()
                state = np.array(state)
                state = np.array(state, dtype=np.float32)

                action = agent.choose_action(state)
                
                #Limiting rpm range to 50-250
                d = 250 - 50
                action = ((action+0.4)/0.75)*d + 50
                if action < 50:
                    action = np.array([action+100])
                elif action > 250:
                    action = np.array([action-100])

                #Send action to PLC
                spool_speed_Model.set_value(int(action))

                time.sleep(0.01)
                #Observe new state
                new_state = [setpoint.get_value(),diameter.get_value(), preform_idler_speed.get_value()]

                #Calculate error
                error = new_state[0] - new_state[1] #Setpoint - Diameter

                #Calculate Accuracy 
                if abs(error)>0.1*new_state[0]:
                    totalPredictions +=1
                else:
                    correctPredictions += 1
                    totalPredictions+= 1

                #Calculate reward
                reward = -abs(error)

                #Remember Experience
                agent.remember(state, action, reward, new_state)

                #Learn from the experience
                agent.learn()

                #Calculate Computation Time
                endtime = time.gmtime()
                endtime_1 = time.time()
                duration = (endtime_1) - (start_time_1)

                #Writing to CSV and Terminal
                print("Setpoint| %.2f"% state[0],"|Diameter| %.2f" % state[1],"RPM from Model %.2f"%int(action) ,"|action_raw| %.2f" % agent.choose_action(state), "|Runtime %.3f"% duration)
                csv_writer.writerow([s,int(action),rpm_from_plc,state[0],state[1],state[0]-state[1],state[2],actual_speed.get_value()])
                csv_file.flush() 

            except KeyboardInterrupt:
                print("Keyboard interrupt detected Exiting...")
                accuracy = (correctPredictions/totalPredictions)*100
                print('Accuracy Score: %.2f' % accuracy, '%')
                break
    finally:
        client.disconnect()
