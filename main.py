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
from model import Model
import train_agent
from utils import *
import pygame
import torch
import Agent
import torch.nn as nn
import torch.optim as optim

# DEFINES AND INITIALIZATIONS
# ----------------------------------------------------------------------------
# Number of sensors in observations
sensor_count = 29

# Number of availiable actions
action_count = 3

# Number of demonstations that the expert preforms
expert_demonstration_count = 1#3

# Episode count of dagger step
dagger_episode_count = 20

# Number of steps per expert iteration
expert_steps = 500#4000

# Number of steps per dagger iteration
dagger_steps = 500#4000

# Number of epochs
epoch_count = 100

# Batch size
batch_size = 32

# If track selection is done manually
manual_reset = False

# If wheel or keyboard is used
using_steering_wheel = True

# FILL HERE IF AUTOMATIC DRIVING
automatic = True

running = True

# All observations and their corresponding actions are stored here
observations_all = [] #= np.zeros((0, sensor_count))
actions_all = [] #= np.zeros((0, action_count))

# Initialize the input interface
# interface = interface.Interface(using_steering_wheel)

# # Create the expert
# expert = expert.Expert(interface, automatic=automatic)

# Initialize Pygame for capturing key events####################################
pygame.init()
screen = pygame.display.set_mode((600, 400))
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
    # Start torcs
    env = gym.make("CarRacing-v3", render_mode="human") #env = gym.TorcsEnv(manual=manual_reset)

    # Observations and actions for this iteration are stored here
    observation_list = []
    action_list = []

    # Expert demonstration
    print("#" * 100)
    print("# Episode: %d start" % episode)
    for i in range(expert_steps):
        # If first iteration, get observation and action
        if i == 0:
            action = np.array([0.0, 0.0, 0.0])#act = env.act
            observation, info = env.reset()#obs = env.obs

        # Get the action from the expert
        handle_keys()#--> mi aggiorna #action act = expert.get_expert_act(act, obs) 

        # Normalize the observation and add it to list of observations
        #obs.normalize_obs()
        # obs_list = obs.get_obs(angle=True, gear=True, rpm=True,
        #                        speedX=True, speedY=True, track=True,
        #                        trackPos=True, wheelSpinVel=True)
        gray_observation = np.dot(observation[...,:3], [0.2126, 0.7152, 0.0722])
        #convert the image to a torch tensor
        gray_observation = torch.tensor(gray_observation[np.newaxis, :, :], dtype=torch.float32)  # Add channel and convert to tensor
        observation_list.append(gray_observation)
        # print(f'gray obs type: {type(gray_observation)} and dimesions: {gray_observation.shape} ') 
        # print(f'action type: {type(action)} and dimesions: {action.size} ') #action is an np array 

        # Normalize the act and add it to list of actions
        # Important to un-normalize the act before sending it to torcs
        #act.normalize_act()
        #act_list = act.get_act(gas=True, gear=True, steer=True)
        action_list.append(torch.from_numpy(action).type(torch.float32))
        #act.un_normalize_act()
        #print(f'action_list type: {type(action_list)} and size {len(action_list)} and action_list[0] type: {type(action_list[0])}')
        print(f'action_list type: {type(action)} and agent_action shape: {action.shape}')

        # Execute the action and get the new observation
        print(action)
        observation, reward, terminated, truncated, info = env.step(action)#obs = env.step(act)
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
        observations_all.append(observation) #= np.concatenate([observations_all, np.reshape(observation, (1, sensor_count))], axis=0)
        print(f'observation_all lenght: {len(observations_all)} and shape of the elements: {observations_all[0].shape}')
        # Concatenate all actions into array of arrays
        actions_all.append(action_made) #=np.concatenate([actions_all, np.reshape(action_made, (1, action_count))], axis=0)
        print(f'action_all lenght: {len(actions_all)} and shape of the elements: {actions_all[0].shape}')

# --------------------------------------------------------------------------------

episode_rewards = []
running = True

# Create the learning agent

model = Agent.Agent(name='model', input_num=observations_all[0].size,
                    output_num=actions_all[0].size)

print("Agent Created")

# Train the model with the observations and actions availiable 
# criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
model.train_model(observations_all, actions_all, n_epoch=epoch_count,
             batch=batch_size)

model.save("dagger_models/model_0.pth")
model_number = 0
old_model_number = 0

print("Agent Trained")

# DAGGER STEP
# ----------------------------------------------------------------------------
# Run the agent and aggregate new data produced by the expert
beta_i = 0.9

for episode in range(dagger_episode_count):
    # Observations and actions for this iteration are stored here
    observation_list = []
    action_list = []
    curr_beta = beta_i ** episode
    episod_reward = 0

    model.load("dagger_models/model_{}.pth".format(model_number))

    if model_number != old_model_number:
        print("Using model : {}".format(episode))
        print("Beta value: {}".format(curr_beta))
        old_model_number = model_number

    

    # Restart the game for every iteration
    env = gym.make("CarRacing-v3", render_mode="human") #env = gym.TorcsEnv(manual=manual_reset)

    print("#" * 100)
    print("# Episode: %d start" % episode)
    
    for i in range(dagger_steps):
        # If first iteration, get observation and action
        if i == 0:
            action = np.array([0.0, 0.0, 0.0])#act = env.act
            observation, info = env.reset()#obs = env.obs

        # If quit key is pressed, prematurely end this run
        #if interface.check_key(pygame.KEYDOWN, pygame.K_q):
            #break
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        if running == False:
            break

        # Get the action that the expert would take
        handle_keys()#new_act = expert.get_expert_act(act, obs)
        #new_act.normalize_act()
        #new_act_list = new_act.get_act(gas=True,
                                       #gear=True,
                                       #steer=True)
        action_list.append(torch.from_numpy(action).type(torch.float32))

        # Normalize the observation and add it to list of observations
        # obs.normalize_obs()
        # obs_list = obs.get_obs(angle=True, gear=True, rpm=True,
        #                        speedX=True, speedY=True, track=True,
        #                        trackPos=True, wheelSpinVel=True)
        gray_observation = np.dot(observation[...,:3], [0.2126, 0.7152, 0.0722])
        #convert the image to a torch tensor
        gray_observation = torch.tensor(gray_observation[np.newaxis, :, :], dtype=torch.float32)  # Add channel and convert to tensor
        observation_list.append(gray_observation)
        
        #ACTION FROM THE AGENT 
        # Normalize the act and add it to list of actions
        # Important to un-normalize the act before sending it to torcs
        agent_action = model.predict(gray_observation).reshape(3,)
        #act.set_act(act_list, gas=True, gear=True, steer=True)
        #act.un_normalize_act()

        # calculate linear combination of expert and network policy
        print(f'expert action: {action}')
        print(f'agent action: {agent_action}')
        pi = curr_beta * action + (1 - curr_beta) * agent_action#.flatten()
        print(f'pi {pi} type: {type(agent_action)} and pi shape: {pi.shape}')






        #print(f'agent_action {agent_action} type: {type(agent_action)} and agent_action shape: {agent_action.shape}')
        # Execute the action and get the new observation
          #agent_action = agent_action.reshape(3,)
        #print(f'agent_action {agent_action} type: {type(agent_action)} and agent_action shape: {agent_action.shape}')
        observation, reward, terminated, truncated, info = env.step(pi)#obs = env.step(act)
        episod_reward += reward
        env.render()


    env.close()

    # Summarize the observations and corresponding actions
    for observation, action_made in zip(observation_list, action_list):
        # Concatenate all observations into array of arrays
        observations_all.append(observation)#observations_all = np.concatenate([observations_all, np.reshape(observation, (1, sensor_count))], axis=0)

        # Concatenate all actions into array of arrays
        actions_all.append(action_made) #=np.concatenate([actions_all, np.reshape(action_made, (1, action_count))], axis=0)


    # Train the model with the aggregated observations and actions
    print("Training the model with the aggregated observations and actions...")
    model.train_model(observations_all, actions_all, n_epoch=epoch_count,
             batch=batch_size)
    model_number += 1
    path = r'C:\Users\39388\Desktop\Dagger\Code\dagger-main\Trained_models\model{}.pth'.format(model_number) 
    model.save(path)
    

    
    episode_rewards.append(episod_reward)



