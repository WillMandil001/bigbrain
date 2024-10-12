import time
import torch
import pygame
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

from IPython.display import clear_output, display

class Brain:
    def __init__(self, shape):
        # self.state = torch.randint(0, 10, tuple(shape), dtype=torch.uint8)
        # self.threshold = torch.randint(0, 10, tuple(shape), dtype=torch.uint8)
        # self.connections_and_strength = [torch.randint(0, 10, tuple(shape), dtype=torch.uint8),
        #                                  torch.randint(0, 10, tuple(shape), dtype=torch.uint8),
        #                                  torch.randint(0, 10, tuple(shape), dtype=torch.uint8),
        #                                  torch.randint(0, 10, tuple(shape), dtype=torch.uint8)]

        # self.state = torch.zeros(tuple(shape), dtype=torch.float16)
        # self.threshold = torch.zeros(tuple(shape), dtype=torch.float16)
        # self.connections_and_strength = [torch.zeros(tuple(shape), dtype=torch.float16),
        #                                  torch.zeros(tuple(shape), dtype=torch.float16),
        #                                  torch.zeros(tuple(shape), dtype=torch.float16),
        #                                  torch.zeros(tuple(shape), dtype=torch.float16)]

        self.state = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.8],
                                   [0.0, 0.0, 0.0, 0.0, 0.0, 0.8],
                                   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float16)

        self.threshold = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                       [0.5, 0.5, 0.3, 0.3, 0.3, 0.3],  # minumum threshold is 1.0!!! this ensures the sumultanious pulsing works!
                                       [0.0, 0.0, 0.0, 0.3, 0.0, 0.3],
                                       [0.0, 0.0, 0.0, 0.3, 0.3, 0.3]], dtype=torch.float16)

        self.connections_and_strength = [torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],# right # existance = connection, value = weight 
                                                       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                       [0.0, 0.0, 0.0, 0.4, 0.4, 0.0]], dtype=torch.float16),
                                    torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                  [0.1, 0.1, 0.1, 0.4, 0.4, 0.4],# left
                                                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float16),
                                    torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],# up
                                                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.4],
                                                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.4]], dtype=torch.float16),
                                    torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                  [0.0, 0.0, 0.0, 0.4, 0.0, 0.0],# down
                                                  [0.0, 0.0, 0.0, 0.4, 0.0, 0.0],
                                                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float16),]

        self.history = []

        self.config = {
            "min_output": 0,
            "max_output": 1.0,
            'history_length': 2,
            "p_spontaneous": 0.001,
            "learning_rate": 1, # should be an integer
            "normalisation_val": 0.1,
            "hebbian_learning_val": 0.01
        }

        # make sure to add a tensor of ones to the list of tensors
        self.connections_and_strength.append(torch.ones_like(self.connections_and_strength[0]))
        stacked_tensors = torch.stack(self.connections_and_strength)
        self.strength = torch.max(stacked_tensors, dim=0)[0]

        # Enable interactive mode
        plt.ion()
        self.fig, ax = plt.subplots(3,4)
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))


    def step(self, input):
        self.state[:, 0] = torch.tensor(input)    # add the input to the first layer of the state

        self.state = self.move_state(self.state, self.connections_and_strength, self.threshold)
        output = self.state[:, -1]    # the last layer of the state is the output

        output = (output - self.config['min_output']) / (self.config['max_output'] - self.config['min_output'])        # scale the output to the range of the action space with min max normalization

        # learning steps
        # self.hebbian_learning_step()
        # self.synaptic_weight_normalisation()

        return output

    def hebbian_learning_step(self):
        if len(self.history) < 2: return

        prev_state = self.history[-2]
        current_state = self.history[-1]

        # roll the inverse direction to get the previous state
        state_right_new = torch.roll(current_state, shifts=-1, dims=1)
        state_left_new = torch.roll(current_state, shifts=1, dims=1)
        state_up_new = torch.roll(current_state, shifts=1, dims=0)
        state_down_new = torch.roll(current_state, shifts=-1, dims=0)
        state_right_new[:, -1] = 0.0
        state_left_new[:, 0] = 0.0
        state_up_new[0, :] = 0.0
        state_down_new[-1, :] = 0.0

        neuron_passed_right = (prev_state != 0) & (state_right_new != 0)
        neuron_passed_left  = (prev_state != 0) & (state_left_new  != 0)
        neuron_passed_up    = (prev_state != 0) & (state_up_new    != 0)
        neuron_passed_down  = (prev_state != 0) & (state_down_new  != 0)

        self.connections_and_strength[0][neuron_passed_right] += self.config['hebbian_learning_val'] 
        self.connections_and_strength[1][neuron_passed_left]  += self.config['hebbian_learning_val']  
        self.connections_and_strength[2][neuron_passed_up]    += self.config['hebbian_learning_val']  
        self.connections_and_strength[3][neuron_passed_down]  += self.config['hebbian_learning_val'] 

        # clip the range to min and max
        self.connections_and_strength[0] = torch.clamp(self.connections_and_strength[0], self.config['min_output'], self.config['max_output'])
        self.connections_and_strength[1] = torch.clamp(self.connections_and_strength[1], self.config['min_output'], self.config['max_output'])
        self.connections_and_strength[2] = torch.clamp(self.connections_and_strength[2], self.config['min_output'], self.config['max_output'])
        self.connections_and_strength[3] = torch.clamp(self.connections_and_strength[3], self.config['min_output'], self.config['max_output'])

    def synaptic_weight_normalisation(self):
        for i in range(len(self.connections_and_strength)):
            self.connections_and_strength[i] /= self.config["normalisation_val"]

    def learn_step(self, input, target):
        # we want to implement:
        # 1. STDP: Spike-timing-dependent plasticity.
        # 2. Homeostasis: Keep the average firing rate of the neurons at a certain level.
        # 3. Inhibition: Inhibit neurons that are too active.
        # 4. Spontaneous firing: Add a small probability of spontaneous firing.
        # 5. Learning from input-target error.
        # 6. Learning from world model prediction error. 

        # hebbian learning:
        pass

    def move_state(self, state, connections_and_strength, threshold):
        # apply thresholding
        new_state = torch.zeros_like(state, dtype=state.dtype)
        indices = state >= threshold
        new_state[indices] = state[indices] - threshold[indices]
        new_state[new_state != 0] = 1

        state_right = new_state*connections_and_strength[0]
        state_left  = new_state*connections_and_strength[1]
        state_up    = new_state*connections_and_strength[2]
        state_down  = new_state*connections_and_strength[3]

        state_right_new = torch.roll(state_right, shifts=1, dims=1)    # shift state_right right by 1
        state_left_new = torch.roll(state_left, shifts=-1, dims=1)    # shift state_left left by 1
        state_up_new = torch.roll(state_up, shifts=-1, dims=0)    # shift state_up up by 1
        state_down_new = torch.roll(state_down, shifts=1, dims=0)    # shift state_down down by 1

        state_right_new[:, 0] = 0
        state_left_new[:, -1] = 0
        state_up_new[-1, :]   = 0
        state_down_new[0, :]  = 0

        new_state = state_left_new + state_right_new + state_up_new + state_down_new

        # spontaneous firing
        if self.config['p_spontaneous'] > 0:
            spontaneous_firing = torch.rand_like(new_state, dtype=torch.float32) < self.config['p_spontaneous']
            new_state[spontaneous_firing] += self.strength[spontaneous_firing]

        if len(self.history) < self.config['history_length']:  self.history.append(new_state)
        else:
            self.history.pop(0)
            self.history.append(new_state)

        # clip the state to the range of the max output
        new_state = torch.clamp(new_state, self.config['min_output'], self.config['max_output'])

        return new_state

    def print_state(self, state, time_step):
        # Clear the previous plot
        self.ax1.cla()
        self.ax2.cla()
        self.ax3.cla()

        state_nump = state.numpy()
        self.ax1.set_xticks(ticks=range(state_nump.shape[1]), labels=range(state_nump.shape[1]))    # Remove axis ticks
        self.ax1.set_yticks(ticks=range(state_nump.shape[0]), labels=range(state_nump.shape[0]))
        self.ax1.imshow(state_nump, cmap='gray', aspect='equal')

        # put the value of each pixel in the image
        for i in range(state_nump.shape[0]): 
            for j in range(state_nump.shape[1]):  self.ax1.text(j, i, f'{state_nump[i, j]:.1f}', color='red', ha='center', va='center')
        self.ax1.set_title("State")

        # plot the weights
        state_nump = self.connections_and_strength[0].numpy()
        self.ax2.set_xticks(ticks=range(state_nump.shape[1]), labels=range(state_nump.shape[1]))    # Remove axis ticks
        self.ax2.set_yticks(ticks=range(state_nump.shape[0]), labels=range(state_nump.shape[0]))
        self.ax2.imshow(state_nump, cmap='gray', aspect='equal')

        # put the value of each pixel in the image
        for i in range(state_nump.shape[0]): 
            for j in range(state_nump.shape[1]):  self.ax2.text(j, i, f'{state_nump[i, j]:.1f}', color='red', ha='center', va='center')
        self.ax2.set_title("Weights")

        # plot the threshold
        state_nump = self.threshold.numpy()
        self.ax3.set_xticks(ticks=range(state_nump.shape[1]), labels=range(state_nump.shape[1]))    # Remove axis ticks
        self.ax3.set_yticks(ticks=range(state_nump.shape[0]), labels=range(state_nump.shape[0]))
        self.ax3.imshow(state_nump, cmap='gray', aspect='equal')

        # put the value of each pixel in the image
        for i in range(state_nump.shape[0]): 
            for j in range(state_nump.shape[1]):  self.ax3.text(j, i, f'{state_nump[i, j]:.1f}', color='red', ha='center', va='center')
        self.ax3.set_title("Threshold")

        # plt.title(f"time step: {time_step}")
        plt.pause(0.01)

env = gym.make("CartPole-v1", render_mode="human")
observation, info = env.reset(seed=42)

# Create the brain
brain = Brain(shape=[4, 5])

for time_step in range(1000):
    env.render()

    action = env.action_space.sample()
    action = brain.step(observation)

    # convert action to binary single value for the environment
    action = round(action[0].item())

    observation, reward, terminated, truncated, info = env.step(action)

    brain.print_state(brain.state, time_step)

    if terminated or truncated:
        observation, info = env.reset()

    time.sleep(0.02)

env.close()
plt.ioff()
plt.show()