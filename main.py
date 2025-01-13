from __future__ import print_function
import argparse
from pyglet.window import key
import gymnasium as gym 
import numpy as np
import pickle
import os
from datetime import datetime
import gzip
import json
#from model import Model
#import train_agent
#from utils import *
import pygame
import torch
import Agent
import torch.nn as nn
import torch.optim as optim
from Utils import find_last_episode

# DEFINES AND INITIALIZATIONS
# ----------------------------------------------------------------------------
# Number of demonstations that the expert preforms
expert_demonstration_count = 3

# Episode count of dagger step
dagger_episode_count = 20

# Number of steps per expert iteration
expert_steps = 4000

# Number of steps per dagger iteration
dagger_steps = 1000#4000

# Number of epochs
epoch_count = 20

# Batch size
batch_size = 128


running = True# A cosa mi serve

# All observations and their corresponding actions are stored here
observations_all = [] 
actions_all = [] 




# Initialize Pygame for capturing key events####################################
pygame.init()
#screen = pygame.display.set_mode((600, 400))
pygame.display.set_caption("Car Racing Control")

# Main function to handle keyboard input
def handle_keys():
    global action
    keys = pygame.key.get_pressed()

    # Steering (left/right arrows), Acceleration (up arrow), Braking (down arrow)
    action[0] = -1.0 if keys[pygame.K_LEFT] else (1.0 if keys[pygame.K_RIGHT] else 0.0)
    action[1] = 1.0 if keys[pygame.K_UP] else 0.0
    action[2] = 0.2 if keys[pygame.K_DOWN] else 0.0

##############################################################################

# EXPERT DEMONSTRATION
# ----------------------------------------------------------------------------
print("Expert Demonstration")
for episode in range(expert_demonstration_count):

    # If the expert dataset is already available there is no need of this part
    if os.path.exists('./Data'):
        break

    # Initializing the envirionment 
    env = gym.make("CarRacing-v3", render_mode="human") 

    # Observations and actions for this iteration are stored here
    observation_list = []
    action_list = []

    # Expert demonstration
    print("#" * 100)
    print(f"# Episode: {episode} start")
    for i in range(expert_steps):
        # If first iteration, get observation and action
        if i == 0:
            action = np.array([0.0, 0.0, 0.0])
            observation, info = env.reset()

        # Get the action from the expert
        handle_keys()#--> updating action, global variable defined in def handle_keys()
        #print(f'obs type: {type(observation)} and dimesions: {observation.shape} ') 
        gray_observation = np.dot(observation[...,:3], [0.2126, 0.7152, 0.0722])
        #convert the image to a torch tensor
        gray_observation = torch.tensor(gray_observation[np.newaxis, :, :], dtype=torch.float32)  # Add channel and convert to tensor
        observation_list.append(gray_observation)
        #print(f'gray obs type: {type(gray_observation)} and dimesions: {gray_observation.shape} ') 
        #print(f'action type: {type(action)} and dimesions: {action.size} ') #action is an np array 

        action_list.append(torch.from_numpy(action).type(torch.float32))
        #print(f'action_list type: {type(action_list)} and size {len(action_list)} and action_list[0] type: {type(action_list[0])}')
        #print(f'action_list type: {type(action)} and agent_action shape: {action.shape}')

        #print(action)
        # Execute the action and get the new observation
        observation, reward, _, _, info = env.step(action)
        env.render()

        # print(f'obs  type: {type(observation)} and dimesions: {observation.shape} ')  
        # print(f'obs_list type: {type(observation_list)} and size {len(observation_list)}')
        # print(f"step: {i}")
        # Process Pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        if running == False:
            break
    # Exit torcs
    env.close()
    pygame.quit()

    # ----------------------------------------------------------------------------
    # Summarizing the demonstration
    print('Packing expert data into arrays...')
    for observation, action_made in zip(observation_list, action_list):
        # Concatenate all observations into array of arrays
        observations_all.append(observation) 
        print(f'observation_all lenght: {len(observations_all)} and shape of the elements: {observations_all[0].shape}')
        # Concatenate all actions into array of arrays
        actions_all.append(action_made) 
        print(f'action_all lenght: {len(actions_all)} and shape of the elements: {actions_all[0].shape}')

# --------------------------------------------------------------------------------
#Saving all the expert dimostrations 
# Save both lists in a dictionary
if not os.path.exists('./Data'):
    os.mkdir('./Data')
    #print(f'observations_all  type: {type(observations_all)} and dimesions: {len(observations_all)}')
    #print(f'observations_all[0]  type: {type(observations_all[0])} and dimesions: {observations_all[0].shape} ')    
    torch.save(observations_all, './Data/Expert_observations_all.pt')  #Serialized into binary format using pickle
    torch.save(actions_all, './Data/Expert_actions_all.pt') 

#Loading expert dimostration
observations_all = torch.load('./Data/Expert_observations_all.pt')
actions_all = torch.load('./Data/Expert_actions_all.pt')
#print(f'observations_all  type: {type(observations_all)} and dimesions: {len(observations_all)} ')
#print(f'observations_all[0]  type: {type(observations_all[0])} and dimesions: {observations_all[0].shape} ')  
#---------------------------------------------------------------------------------

print('Expert dataset is available')

episode_rewards = []
running = True

# Create the learning agent
model = Agent.Agent(name='model', input_num=observations_all[0].size,
                    output_num=actions_all[0].size)
print("Agent Created")


#checking if we already trained the agent with expert informations .
if not os.path.exists('./Models'):
    os.mkdir('./Models')

    # Train the model with the observations and actions availiable 
    model.train_model(observations_all, actions_all, n_epoch=epoch_count,
                batch=batch_size, session_id = '1st_Training')
    
    print("Agent Trained")

    model.save("./Models/model_0.pth")
    print("Agent Saved")

else:
    print("Initial Agent already trained")




# DAGGER STEP
# ----------------------------------------------------------------------------
# Run the agent and aggregate new data produced by the expert
beta_i = 0.9


# Directory containing saved models
model_directory = r'Models'

# Check the last saved model
start_episode = find_last_episode(model_directory)

model_number = start_episode
old_model_number = start_episode

#If we are restarting the Dagger loop after a break, we need to load the last saved dataset with all the (Obs, Act) pairs 
# if os.path.exists('./Data/Exp_obs_&_correction.pt'):
#     observations_all = torch.load(r'./Data/Exp_obs_&_correction.pt')
#     actions_all = torch.load(r'./Data/Exp_act_&_correction.pt')

print(f"The Dagger Loop will restart from episode: {start_episode}")

for episode in range(start_episode, dagger_episode_count):


    # Observations and actions for this iteration are stored here
    observation_list = []
    action_list = []
    curr_beta = beta_i ** episode
    episod_reward = 0


    model.load("./Models/model_{}.pth".format(model_number))   

    if model_number != old_model_number:
        print(f"Using model : {episode}")
        print(f"Beta value: {curr_beta}")
        old_model_number = model_number

    

    # Restart the game for every iteration
    env = gym.make("CarRacing-v3", render_mode="human") 

    print("#" * 100)
    print(f"# Episode: {episode} start")
    
    for i in range(dagger_steps):
        # If first iteration, get observation and action
        if i == 0:
            action = np.array([0.0, 0.0, 0.0])#act = env.act
            observation, info = env.reset()#obs = env.obs

        # If quit key is pressed, prematurely end this run
        for event in pygame.event.get():
            if event.type == pygame.QUIT:  
                running = False
        if running == False:
            break

        # Get the action that the expert would take
        handle_keys()
   
        action_list.append(torch.from_numpy(action).type(torch.float32))

        gray_observation = np.dot(observation[...,:3], [0.2126, 0.7152, 0.0722])#image to black and withe 
        #convert the image to a torch tensor
        gray_observation = torch.tensor(gray_observation[np.newaxis, :, :], dtype=torch.float32)  # Add channel and convert to tensor
        observation_list.append(gray_observation)
        
        #ACTION FROM THE AGENT 
        agent_action = model.predict(gray_observation).reshape(3,)
 
        # calculate linear combination of expert and network policy
        print(f'expert action: {action}')
        print(f'agent action: {agent_action}')
        pi = curr_beta * action + (1 - curr_beta) * agent_action
        print(f'pi {pi} type: {type(agent_action)} and pi shape: {pi.shape}')
        #print(f'agent_action {agent_action} type: {type(agent_action)} and agent_action shape: {agent_action.shape}')

        # Execute the action and get the new observation
        observation, reward, terminated, truncated, info = env.step(pi)
        episod_reward += reward
        env.render()


    env.close()

    # Summarize the observations and corresponding actions
    for observation, action_made in zip(observation_list, action_list):
        # Concatenate all observations into array of arrays
        observations_all.append(observation)

        # Concatenate all actions into array of arrays
        actions_all.append(action_made) 

    torch.save(observations_all, './Data/Exp_obs_&_correction.pt')  #Serialized into binary format using pickle
    torch.save(actions_all, './Data/Exp_act_&_correction.pt') 



    # Train the model with the aggregated observations and actions
    print("Training the model with the aggregated observations and actions...")
    model_number += 1
    model.train_model(observations_all, actions_all, n_epoch=epoch_count,
             batch=batch_size, session_id= f'model_{model_number}_Loss')
    path = r'.\Models\model_{}.pth'.format(model_number) 
    model.save(path)
    

    
    episode_rewards.append(episod_reward)



