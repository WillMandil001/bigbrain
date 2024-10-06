import time
import torch
import pygame
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

from IPython.display import clear_output, display

class Brain:
    def __init__(self, shape):
        self.state = torch.randint(0, 10, tuple(shape), dtype=torch.uint8)

        self.threshold = torch.randint(0, 10, tuple(shape), dtype=torch.uint8)

        self.connections_and_strength = [torch.randint(0, 10, tuple(shape), dtype=torch.uint8),
                                         torch.randint(0, 10, tuple(shape), dtype=torch.uint8),
                                         torch.randint(0, 10, tuple(shape), dtype=torch.uint8),
                                         torch.randint(0, 10, tuple(shape), dtype=torch.uint8)]

        self.history = []

        self.config = {
            "min_output": 0,
            "max_output": 256,
            'history_length': 10,
            "p_spontaneous": 0.1,
            "learning_rate": 0.01
        }

        # make sure to add a tensor of ones to the list of tensors
        self.connections_and_strength.append(torch.ones_like(self.connections_and_strength[0]))
        stacked_tensors = torch.stack(self.connections_and_strength)
        self.strength = torch.max(stacked_tensors, dim=0)[0]

        # Enable interactive mode
        plt.ion()
        self.fig, (self.ax1, self.ax2) = plt.subplots(2,1)

    def step(self, input):
        # scale the input to the range of the state with min max normalization (input is -1, 1)
        input = (input[0] + 1) * 128

        self.state[:, 0] = torch.tensor(input)    # add the input to the first layer of the state

        self.state = self.move_state(self.state, self.connections_and_strength, self.threshold)
        output = self.state[:, -1]    # the last layer of the state is the output

        output = (output - self.config['min_output']) / (self.config['max_output'] - self.config['min_output'])        # scale the output to the range of the action space with min max normalization

        # learning steps
        self.hebbian_learning_step(learning_rate=self.config["learning_rate"])

        return output

    def hebbian_learning_step(self, learning_rate=0.01):
        # Use the current state after thresholding
        S = self.state.clone()

        # Get the pre-synaptic activities by shifting the state tensor
        S_pre_left = torch.roll(S, shifts=-1, dims=1)
        S_pre_right = torch.roll(S, shifts=1, dims=1)
        S_pre_up = torch.roll(S, shifts=-1, dims=0)
        S_pre_down = torch.roll(S, shifts=1, dims=0)
    
        S_pre_left[:, -1] = 0  # Handle left boundary
        S_pre_right[:, 0] = 0  # Handle right boundary
        S_pre_up[-1, :] = 0  # Handle top boundary
        S_pre_down[0, :] = 0  # Handle bottom boundary

        # Compute the weight updates based on Hebbian learning rule
        delta_W_left = learning_rate * S * S_pre_left
        delta_W_right = learning_rate * S * S_pre_right
        delta_W_up = learning_rate * S * S_pre_up
        delta_W_down = learning_rate * S * S_pre_down

        # Update the weights for each direction
        self.connections_and_strength[0] += delta_W_right.round().to(torch.uint8)  # Right connections
        self.connections_and_strength[1] += delta_W_left.round().to(torch.uint8)   # Left connections
        self.connections_and_strength[2] += delta_W_up.round().to(torch.uint8)     # Up connections
        self.connections_and_strength[3] += delta_W_down.round().to(torch.uint8)   # Down connections

        # Update the weights for each direction, clamp to avoid overflow
        self.connections_and_strength[0] = torch.clamp(self.connections_and_strength[0].to(torch.int32) + delta_W_right.round().to(torch.int32), min=self.config['min_output'], max=self.config['max_output']).to(torch.uint8)  # Right connections
        self.connections_and_strength[1] = torch.clamp(self.connections_and_strength[1].to(torch.int32) + delta_W_left.round().to(torch.int32), min=self.config['min_output'], max=self.config['max_output']).to(torch.uint8)   # Left connections
        self.connections_and_strength[2] = torch.clamp(self.connections_and_strength[2].to(torch.int32) + delta_W_up.round().to(torch.int32), min=self.config['min_output'], max=self.config['max_output']).to(torch.uint8)     # Up connections
        self.connections_and_strength[3] = torch.clamp(self.connections_and_strength[3].to(torch.int32) + delta_W_down.round().to(torch.int32), min=self.config['min_output'], max=self.config['max_output']).to(torch.uint8)   # Down connections


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
        new_state[new_state != 0] = 1.0

        state_right = new_state*connections_and_strength[0]
        state_left  = new_state*connections_and_strength[1]
        state_up    = new_state*connections_and_strength[2]
        state_down  = new_state*connections_and_strength[3]

        state_right_new = torch.roll(state_right, shifts=1, dims=1)    # shift state_right right by 1
        state_left_new = torch.roll(state_left, shifts=-1, dims=1)    # shift state_left left by 1
        state_up_new = torch.roll(state_up, shifts=-1, dims=0)    # shift state_up up by 1
        state_down_new = torch.roll(state_down, shifts=1, dims=0)    # shift state_down down by 1

        state_right_new[:, 0] = 0.0
        state_left_new[:, -1] = 0.0
        state_up_new[-1, :] = 0.0
        state_down_new[0, :] = 0.0

        new_state = state_left_new + state_right_new + state_up_new + state_down_new

        # spontaneous firing
        if self.config['p_spontaneous'] > 0:
            spontaneous_firing = torch.rand_like(new_state, dtype=torch.float32) < self.config['p_spontaneous']
            new_state[spontaneous_firing] += self.strength[spontaneous_firing]

        if len(self.history) < self.config['history_length']:
            self.history.append(new_state)
        else:
            self.history.pop(0)
            self.history.append(new_state)

        return new_state

    def print_state(self, state, time_step):
        # Clear the previous plot
        self.ax1.cla()
        self.ax2.cla()

        state_nump = state.numpy()
        self.ax1.set_xticks(ticks=range(state_nump.shape[1]), labels=range(state_nump.shape[1]))    # Remove axis ticks
        self.ax1.set_yticks(ticks=range(state_nump.shape[0]), labels=range(state_nump.shape[0]))
        self.ax1.imshow(state_nump, cmap='gray', aspect='equal')

        # put the value of each pixel in the image
        for i in range(state_nump.shape[0]): 
            for j in range(state_nump.shape[1]):  self.ax1.text(j, i, f'{state_nump[i, j]:.2f}', color='red', ha='center', va='center')

        plt.title(f"time step: {time_step}")

        # plot the weights
        state_nump = self.connections_and_strength[0].numpy()
        self.ax2.set_xticks(ticks=range(state_nump.shape[1]), labels=range(state_nump.shape[1]))    # Remove axis ticks
        self.ax2.set_yticks(ticks=range(state_nump.shape[0]), labels=range(state_nump.shape[0]))
        self.ax2.imshow(state_nump, cmap='gray', aspect='equal')

        # put the value of each pixel in the image
        for i in range(state_nump.shape[0]): 
            for j in range(state_nump.shape[1]):  self.ax2.text(j, i, f'{state_nump[i, j]:.2f}', color='red', ha='center', va='center')

        plt.pause(0.01)

env = gym.make("CartPole-v1", render_mode="human")
observation, info = env.reset(seed=42)

# Create the brain
brain = Brain(shape=[4, 10])

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