import numpy as np
import matplotlib.pyplot as plt

def plot_detailed(history_of_spikes, history_of_voltage, history_of_random_spikes, history_of_voltage_threshold, fig, axs, fig2, axs2, num_neurons, weights):
    # plt in a subplot for each of the neurons as a graph of its state for the last 100 timesteps

    for i in range(len(axs)):
        axs[i].cla()

    history_of_spikes_ = np.array(history_of_spikes)
    history_of_voltage_ = np.array(history_of_voltage)
    history_of_random_spikes_ = np.array(history_of_random_spikes)
    history_of_voltage_threshold_ = np.array(history_of_voltage_threshold)

    for neuron_to_plot in range(num_neurons):
        axs[neuron_to_plot].plot(history_of_voltage_[:, neuron_to_plot], label=f'Neuron {1}', c='b')
        axs[neuron_to_plot].plot(history_of_voltage_threshold_[:, neuron_to_plot], label=f'threshold {1}', c='g')

        spike_time = [i for i, x in enumerate(history_of_random_spikes_[:, neuron_to_plot]) if x >= 1]            # plot the input spikes (random spikes)
        spike_numbers = history_of_random_spikes_[:, neuron_to_plot][spike_time]

        axs[neuron_to_plot].vlines(x=spike_time, ymin=0, ymax=max(history_of_voltage_threshold_[:, neuron_to_plot]), color='r', linestyles='dashed')
        for i in range(len(spike_numbers)):  # add the number of spikes as a letter above the spike
            axs[neuron_to_plot].text(spike_time[i], max(history_of_voltage_threshold_[:, neuron_to_plot]), str(spike_numbers[i]), fontsize=12, color='r')

        axs[neuron_to_plot].set_ylabel('voltage')
        # axs[neuron_to_plot].set_xlabel('Time (ms)')
        axs[neuron_to_plot].set_title(f'Neuron {neuron_to_plot}')
        if neuron_to_plot == 0:
            axs[neuron_to_plot].legend()
    
    fig.tight_layout()


    # plot the weights in the last subplot
    axs2.imshow(weights, cmap='gray', aspect='equal')
    # place the value of the neurons voltage over the image
    for i in range(num_neurons):
        for j in range(num_neurons):
            axs2.text(j, i, '{:.2f}'.format(weights[i, j].item()), color='red', ha='center', va='center')
    fig2.tight_layout()
    
    plt.show()
    plt.pause(0.0001)


def plot_simple(v, num_neurons, vt):
    plt.clf()
    plt.subplot(2, 1, 1)

    plt.title('voltage')
    plt.imshow(v.view(1, -1), cmap='gray', aspect='equal')
    for i in range(num_neurons):        # place the value of the neurons voltage over the image
        plt.text(i, 0, '{:.2f}   '.format(v[i].item()), color='red', ha='center', va='center')

    plt.xticks(np.arange(-0.5, num_neurons, step=1))        # set the ticks but shift them to be ever .5 value not 1.0 value
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)        # set tick values to dissapear
    plt.grid(axis='x')        # add the grid to only the horizontal axis

    plt.subplot(2, 1, 2)
    plt.title('voltage threshold')
    plt.imshow(vt.view(1, -1), cmap='gray', aspect='equal')
    for i in range(num_neurons):        # place the value of the neurons voltage over the image
        plt.text(i, 0, '{:.2f}   '.format(vt[i].item()), color='red', ha='center', va='center')
    plt.xticks(np.arange(-0.5, num_neurons, step=1))        # set the ticks but shift them to be ever .5 value not 1.0 value
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)        # set tick values to dissapear
    plt.grid(axis='x')        # add the grid to only the horizontal axis

    plt.show()
    plt.pause(0.0001)
