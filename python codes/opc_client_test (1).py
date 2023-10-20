import sys
sys.path.insert(0, "..")
from opcua import Client
import time
import tensorflow as tf
import numpy as np
from collections import deque

if __name__ == "__main__":
    client = Client("opc.tcp://DESKTOP-A9QNR1L:4990/FactoryTalkLinxGateway1")
    #connect using a user
    # client = Client("opc.tcp://admin@localhost:4840/freeopcua/server/")
    try:
        client.connect()
        model = tf.keras.models.load_model('C:/Users/mit-f/Documents/model2/model1')

        # Client has a few methods to get proxy to UA nodes that
        #  should always be in address space such as Root or Objects
        root = client.get_root_node()
        print("Objects node is: ", root)

        # Node objects have methods to read and write node attributes
        #  as well as browse or populate address space
        print("Children of root are: ", root.get_children())

        # get a specific node knowing its node id
        #var = client.get_node(ua.NodeId(1002, 2))
        setpoint = client.get_node("ns=2;s=[opc_server]setpoint_diameter")
        diameter = client.get_node("ns=2;s=[opc_server]Diameter")
        setpoint_speed = client.get_node("ns=2;s=[opc_server]DC_output")

        setpoint_data = deque(maxlen=32)
        diameter_data = deque(maxlen=32)

        setpoint_data.append(np.zeros(32))
        diameter_data.append(np.zeros(32))

        # print(var.get_value())
        # print("motor_state:", end='')
        # state = input()
        # var.set_value(int(state))
        
        #var.write_value(True)
        #print(state)

        # var =  client.nodes.root.get_child("ns=2;s=[opc_server]Motor_Run")
        # var.write(True)

        #var.get_data_value() # get value of node as a DataValue object
        #var.get_value() # get value of node as a python builtin
        #var.set_value(3.9) # set node value using implicit data type

        # Now getting a variable node using its browse path
        # myvar = root.get_child(["0:Objects", "2:MyObject", "2:MyVariable"])
        # obj = root.get_child(["0:Objects", "2:MyObject"])
        # print("myvar is: ", myvar)
        # print("myobj is: ", obj)
        # print(myvar.get_value())


        # time.sleep(1)

        # #try to write to the server
        # var.set_value(2)
        # time.sleep(5)
        # var.set_value(0)

        while(True):
            try: 
                d = diameter_array.get_value()
                s = setpoint_array.get_value()

                setpoint_data.append(s)
                diameter_data.append(d)

                setpoint_array = np.array(setpoint_data).reshape(-1, 1)
                print('The shape of the set point array is ', setpoint_array.shape())
                diameter_array = np.array(diameter_data).reshape(-1, 1)
                print('The shape of the set point array is ', diameter_array.shape())

                #normalize
                d_means = 0.42954475
                s_means = 0.40073333
                d_std = 0.13086448
                s_std = 0.16299119 

                d_normalized = (diameter_array-d_means)/d_std
                s_normalized = (setpoint_array-s_means)/s_std     
                
                # print(d,d_normalized)
                # print(type(d_normalized))
                # print(type(0.5))
                
                input = np.array([d_normalized,s_normalized])
                predictions = model.predict(input)
                predictions = predictions.squeeze()
                print(predictions)
                print(type(predictions))
                if predictions <= 15:
                    setpoint_speed.set_value(15)
                elif predictions >= 200:
                    setpoint_speed.set_value(200)
                else:
                    setpoint_speed.set_value(float(predictions))

                time.sleep(0.05)
            except KeyboardInterrupt:
                print("Keyboard interrupt detected. Exiting...")
                break



    finally:
        client.disconnect()