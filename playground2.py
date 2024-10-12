import time
import torch
import random
import numpy as np
import matplotlib.pyplot as plt

from IPython.display import clear_output, display

from utils import plot_detailed

class spiking_network:
    def __init__(self, size, device, plotting=False):
        self.num_neurons = size
        self.plotting = plotting
        self.device = device  # Add device attribute

        density = 0.9  # 10% of the connections are non-zero
        self.W      = (torch.rand(self.num_neurons, self.num_neurons, device=self.device) < density).float() * torch.randn(self.num_neurons, self.num_neurons, device=self.device)
        self.W.fill_diagonal_(0)  # remove self-connections
        self.v      = torch.ones(self.num_neurons, device=self.device) * 0.5
        self.vt     = torch.ones(self.num_neurons, device=self.device)
        self.vt_min = torch.ones(self.num_neurons, device=self.device)

        self.spikes = torch.zeros(self.num_neurons, dtype=torch.float32, device=self.device)

        self.dt = 1.0  # time step

        # rate of decay
        self.voltage_threshold_decay_rate = 0.001
        self.voltage_decay_rate = 20.0

        # STDP parameters
        self.tau_plus = 20.0
        self.tau_minus = 20.0
        self.A_plus = 0.01
        self.A_minus = 0.012
        self.max_weight = 1.0  # Maximum weight value for clipping

        # Initialize last spike times and current time
        self.last_spike_time = torch.full((self.num_neurons,), -float('inf'), device=self.device)
        self.time = 0

        # histroy storage
        self.histroy_horizon = 100
        self.history_of_random_spikes, self.history_of_voltage_threshold, self.history_of_voltage, self.history_of_spikes = [], [], [], []

        # setup the probability of spontaneous spiking
        probabilities = torch.tensor([0.8, 0.1, 0.05, 0.04, 0.005, 0.005], device=self.device)
        self.probabilities = probabilities / probabilities.sum()

        self.setup_plotting()

    def forward(self):
        self.spikes += self.spontaneous_spike()  # get the random spikes

        # update the membrane potential
        self.v += torch.matmul(self.W, self.spikes)
        self.v -= (self.v / self.voltage_decay_rate) * self.dt  # decay the membrane potential

        # threshold decay
        self.vt -= self.voltage_threshold_decay_rate
        self.vt[self.vt < 1.0] = 1.0  # clip the membrane potential

        # update the spikes
        spiking_neurons = self.v >= self.vt   # check which neurons have exceeded the spike threshold
        self.vt[spiking_neurons] += 0.1       # increase decaying threshold linearly
        self.v[spiking_neurons] = 0.0         # reset the membrane potential
        self.v = torch.clamp(self.v, min=0.0) # clip the membrane potential

        # store the spikes
        self.spikes = spiking_neurons.float()

        # STDP learning rule
        self.stdp_update(spiking_neurons)

        # Update last spike times
        self.last_spike_time[spiking_neurons] = self.time

        # Store history and plot
        if self.plotting:
            self.store_history(spiking_neurons)
            if self.time % 10 == 0:
                plot_detailed(self.history_of_spikes, self.history_of_voltage, self.history_of_random_spikes, self.history_of_voltage_threshold, self.fig, self.axs, self.fig2, self.axs2, self.num_neurons, self.W)

        # reset the spikes and increment the time
        self.spikes.zero_()
        self.time += 1

        return spiking_neurons

    def spontaneous_spike(self):
        return torch.multinomial(self.probabilities, self.num_neurons, replacement=True)

    def stdp_update(self, spiking_neurons):
        # Vectorized STDP update
        # Create t_post vector
        t_post = torch.full((self.num_neurons,), float('nan'), device=self.device)
        t_post[spiking_neurons] = self.time

        t_pre = self.last_spike_time  # size (num_neurons,)

        # Compute delta_t matrix
        delta_t = t_post.unsqueeze(1) - t_pre.unsqueeze(0)  # size (num_neurons, num_neurons)

        # LTP -  Pre before post: Potentiate synapse
        mask_LTP = (delta_t > 0) & (delta_t <= self.tau_plus)
        delta_w_LTP = self.A_plus * torch.exp(-delta_t / self.tau_plus)
        delta_w_LTP[~mask_LTP] = 0.0

        # LTD - Post before pre: Depress synapse
        mask_LTD = (delta_t < 0) & (delta_t >= -self.tau_minus)
        delta_w_LTD = -self.A_minus * torch.exp(delta_t / self.tau_minus)
        delta_w_LTD[~mask_LTD] = 0.0

        # Total delta_w
        delta_w = delta_w_LTP + delta_w_LTD

        # Zero out diagonal (self-connections)
        delta_w.fill_diagonal_(0.0)

        # Update weights
        self.W += delta_w

        # Clip the weights to prevent them from growing too large
        self.W = torch.clamp(self.W, min=-self.max_weight, max=self.max_weight)

    def setup_plotting(self):
        plt.ion()        # Enable interactive mode
        ncols = 3
        nrows = (self.num_neurons + 2) // ncols
        self.fig, self.axs = plt.subplots(nrows, ncols, figsize=(10, 5))
        self.axs = self.axs.flatten()

        self.fig2, self.axs2 = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))

    def store_history(self, spiking_neurons):
        self.history_of_random_spikes.append(self.spikes.clone().cpu())
        self.history_of_spikes.append(spiking_neurons.clone().cpu())
        self.history_of_voltage.append(self.v.clone().cpu())
        self.history_of_voltage_threshold.append(self.vt.clone().cpu())
        if len(self.history_of_voltage_threshold) > self.histroy_horizon:  # remove the oldest value if the history is too long
            self.history_of_voltage_threshold.pop(0)
            self.history_of_voltage.pop(0)
            self.history_of_spikes.pop(0)
            self.history_of_random_spikes.pop(0)

# build the spiking network
num_neurons = 1000
device = 'cuda' if torch.cuda.is_available() else 'cpu'
spike_network = spiking_network(size=num_neurons, device=device, plotting=False)

# run the simulation
start_time = time.time()
for timestep in range(100):
    print(timestep)
    spiking_neurons = spike_network.forward()
end_time = time.time()

print(f"Simulation ran in {end_time - start_time:.2f} seconds.")

plt.ioff()