import time
import torch
import random
import numpy as np
import matplotlib.pyplot as plt

from IPython.display import clear_output, display

from utils import plot_detailed

class spiking_network:
    def __init__(self, size):
        self.num_neurons = size
        # self.weight_scale = 0.1

        self.w = torch.ones(self.num_neurons) * 0.5   # 
        # self.W = torch.randn(self.num_neurons, self.num_neurons) * weight_scale
        self.v = torch.ones(self.num_neurons) * 0.5

        self.vt     = torch.ones(self.num_neurons)
        self.vt_min = torch.ones(self.num_neurons)

        self.spikes = torch.zeros(self.num_neurons, dtype=torch.int)

        self.dt = 1.0  # time step

        # rate of decay
        self.voltage_threshold_decay_rate = 0.001
        self.voltage_decay_rate = 20.0

        # histroy storage
        self.histroy_horizon = 100
        self.history_of_random_spikes, self.history_of_voltage_threshold, self.history_of_voltage, self.history_of_spikes = [], [], [], []  

        # setup the probability of spontaneous spiking
        probabilities = torch.tensor([0.8, 0.1, 0.05, 0.04, 0.005, 0.005])
        self.probabilities = probabilities / probabilities.sum()

        self.setup_plotting()

    def forward(self):
        self.spikes += self.spontaneous_spike()  # get the random spikes

        # calculate the synaptic input from other neurons
        # self.v += torch.matmul(self.W, self.spikes.float())

        # update the membrane potential
        self.v += (self.w * self.spikes)
        self.v -= (self.v / self.voltage_decay_rate) * self.dt  # decay the membrane potential

        # threshold decay
        self.vt -= self.voltage_threshold_decay_rate
        self.vt[self.vt < 1.0] = 1.0  # clip the membrane potential

        # update the spikes
        spiking_neurons = self.v >= self.vt   # check which neurons have exceeded the spike threshold
        self.vt[spiking_neurons] += 0.1       # increase decaying threshold linearly
        self.v[spiking_neurons] = 0.0         # reset the membrane potential
        self.v = torch.clamp(self.v, min=0.0) # clip the membrane potential

        # Store history and plot
        self.store_history(spiking_neurons)
        plot_detailed(self.history_of_spikes, self.history_of_voltage, self.history_of_random_spikes, self.history_of_voltage_threshold, self.axs, self.num_neurons)

        return spiking_neurons

    def spontaneous_spike(self):
        return torch.multinomial(self.probabilities, self.num_neurons, replacement=True)

    def setup_plotting(self):
        plt.ion()        # Enable interactive mode
        ncols = 3
        nrows = (self.num_neurons + 2) // ncols
        self.fig, self.axs = plt.subplots(nrows, ncols, figsize=(10, 5))
        self.axs = self.axs.flatten()

    def store_history(self, spiking_neurons):
        self.history_of_random_spikes.append(self.spikes.clone())
        self.history_of_spikes.append(spiking_neurons.clone())
        self.history_of_voltage.append(self.v.clone())
        self.history_of_voltage_threshold.append(self.vt.clone())
        if len(self.history_of_voltage_threshold) > self.histroy_horizon:        # remove the oldest value if the history is too long
            self.history_of_voltage_threshold.pop(0)
            self.history_of_voltage.pop(0)
            self.history_of_spikes.pop(0)
            self.history_of_random_spikes.pop(0)

# build the spiking network
num_neurons = 10
spike_network = spiking_network(size=num_neurons)

# run the simulation
for timestep in range(100):
    spiking_neurons = spike_network.forward()    # forward pass
    time.sleep(0.2)

plt.ioff()
plt.show()