import time
import torch
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

from utils import print_state, config
from IPython.display import clear_output, display

class Brain:
    def __init__(self, shape):
        self.shape = shape
        self.config = config

        self.state = torch.zeros(tuple(shape), dtype=torch.float16)
        self.threshold = torch.rand(tuple(shape), dtype=torch.float16)
        self.connections_and_strength = [torch.rand(tuple(shape), dtype=torch.float16) for i in range(4)]
        for arr in self.connections_and_strength: arr[arr > 0.1] = 0.0    # randomly zero out some of the connections_and_strength

        self.history = [self.state, self.state]

        plt.ion()        # Enable interactive mode
        self.plot_height, self.plot_width = 3, 4
        self.fig, self.axes = plt.subplots(self.plot_height, self.plot_width, figsize=(15, 7))

    def step(self, input, reward):
        self.reward_based_hebbian_learning_step(reward)
        self.weight_decay()
        self.threshold_decay()
        self.synaptic_weight_normalisation()
        self.spontaenous_spike()

        self.state[:, 0] = torch.tensor(input)  # add the input to the first layer of the state

        self.state = self.move_state(self.state, self.connections_and_strength, self.threshold)
        output = self.state[:, -1]    # the last layer of the state is the output

        return output

    def threshold_decay(self):
        self.threshold += self.config['threshold_decay_rate'] * self.threshold
        self.threshold = torch.clamp(self.threshold, 0.0, 1.0)        # clamp the threshold at 0.        

    def weight_decay(self):
        for i in range(4):
            self.connections_and_strength[i] -= self.config['strength_decay_rate'] * self.connections_and_strength[i]
            self.connections_and_strength[i] = torch.clamp(self.connections_and_strength[i], 0.0, 1.0)       # clamp the connections_and_strength at 0.0     

    def spontaenous_spike(self):
        spontaneous_firing = torch.rand_like(self.state) < self.config['p_spontaneous']
        self.state[spontaneous_firing] = self.threshold[spontaneous_firing]        #spike at the given connection with the given strength
        self.state = torch.clamp(self.state, self.config['min_output'], self.config['max_output'])        # clip the state to the range of the max output

    def reward_based_hebbian_learning_step(self, reward):
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

        for direction, passes in enumerate([neuron_passed_right, neuron_passed_left, neuron_passed_up, neuron_passed_down]): 
            self.connections_and_strength[direction][passes] += self.config['hebbian_learning_strength_val'] * reward

        # clip the range to min and max
        for i in range(4): 
            self.connections_and_strength[i] = torch.clamp(self.connections_and_strength[i], self.config['min_output'], self.config['max_output'])

        # also reduce the threshold of the neurons that fired
        for direction, passes in enumerate([neuron_passed_right, neuron_passed_left, neuron_passed_up, neuron_passed_down]): 
            self.threshold[passes] -= self.config['hebbian_learning_thresh_val'] * reward
        self.threshold = torch.clamp(self.threshold, 0.0, 1.0)        # clamp the threshold at 0.0

    def synaptic_weight_normalisation(self):
        for i in range(len(self.connections_and_strength)):
            self.connections_and_strength[i] *= (self.config["normalisation_val"] / (self.shape[0] * self.shape[1]))

    def move_state(self, state, connections_and_strength, threshold):
        # apply thresholding
        new_state = torch.zeros_like(state, dtype=state.dtype)
        indices = state >= threshold
        new_state[indices] = state[indices] - threshold[indices]
        new_state[new_state != 0] = 1

        # add a small probability of spontaneous firing
        spontaneous_firing = torch.rand_like(state) < self.config['p_spontaneous']
        new_state[spontaneous_firing] = 1

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

        # clip the state to the range of the max output
        new_state = torch.clamp(new_state, self.config['min_output'], self.config['max_output'])

        self.history.pop(0)
        self.history.append(new_state)

        return new_state

# env = gym.make("CartPole-v1", render_mode="human")
env = gym.make("Pendulum-v1", render_mode="rgb_array")
observation, info = env.reset(seed=42)
max_reward = -16.2736044
reward = 0.0

# Create the brain
brain = Brain(shape=[4, 5])

for time_step in range(10000):
    env.render()

    # observation is between -1 and 1, set it to between 0 and 1
    observation[:2] = (observation[:2] + 1) / 2
    observation[2]  = (observation[2] + 8) / 16
    observation = np.append(observation, 0.0)

    # scale reward to between 0 and 1
    reward = (reward - max_reward) / 100

    action = brain.step(observation, reward)
    action = (action - brain.config['min_output']) / (brain.config['max_output'] - brain.config['min_output'])        # scale the output to the range of the action space with min max normalization
    action = (action[:1] - 0.5) * 4

    observation, reward, terminated, truncated, info = env.step(np.array(action, dtype=np.float32))

    if time_step % 1000 == 0:
        print_state(brain.state, brain.history, brain.threshold, brain.connections_and_strength, brain.plot_height, brain.plot_width, plt, brain.axes)
        print(f"Time step: {time_step}, Reward: {reward}")

    if terminated or truncated:  observation, info = env.reset()

    # time.sleep(0.01)

env.close()
plt.ioff()
plt.show()