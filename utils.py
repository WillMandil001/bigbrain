
config = {
    "min_output": 0,
    "max_output": 1.0,
    'history_length': 2,
    "p_spontaneous": 0.1,
    "learning_rate": 1, # should be an integer
    "normalisation_val": 20.0,
    "hebbian_learning_strength_val": 0.1,
    "hebbian_learning_thresh_val": 0.1,
    "strength_decay_rate": 0.001,
    "threshold_decay_rate": 0.0001,
}

def print_state(state, history, threshold, connections_and_strength, plot_height, plot_width, plt, axes):
    # Clear the previous plot
    for i in range(plot_height):
        for j in range(plot_width):
            axes[i, j].cla()

    # state
    state_nump = state.numpy()
    axes[0, 0].set_xticks(ticks=range(state_nump.shape[1]), labels=range(state_nump.shape[1]))    # Remove axis ticks
    axes[0, 0].set_yticks(ticks=range(state_nump.shape[0]), labels=range(state_nump.shape[0]))
    axes[0, 0].imshow(state_nump, cmap='gray', aspect='equal', vmin=0, vmax=1)
    for i in range(state_nump.shape[0]): 
        for j in range(state_nump.shape[1]):  axes[0, 0].text(j, i, f'{state_nump[i, j]:.1f}', color='red', ha='center', va='center')
    axes[0, 0].set_title("State")

    # previous state
    state_nump = history[-2].numpy()
    axes[0, 1].set_xticks(ticks=range(state_nump.shape[1]), labels=range(state_nump.shape[1]))    # Remove axis ticks
    axes[0, 1].set_yticks(ticks=range(state_nump.shape[0]), labels=range(state_nump.shape[0]))
    axes[0, 1].imshow(state_nump, cmap='gray', aspect='equal', vmin=0, vmax=1)
    for i in range(state_nump.shape[0]): 
        for j in range(state_nump.shape[1]):  axes[0, 1].text(j, i, f'{state_nump[i, j]:.1f}', color='red', ha='center', va='center')
    axes[0, 1].set_title("State")

    # plot the threshold
    state_nump = threshold.numpy()
    axes[1, 0].set_xticks(ticks=range(state_nump.shape[1]), labels=range(state_nump.shape[1]))    # Remove axis ticks
    axes[1, 0].set_yticks(ticks=range(state_nump.shape[0]), labels=range(state_nump.shape[0]))
    axes[1, 0].imshow(state_nump, cmap='gray', aspect='equal', vmin=0, vmax=1)
    for i in range(state_nump.shape[0]): 
        for j in range(state_nump.shape[1]):  axes[1, 0].text(j, i, f'{state_nump[i, j]:.2f}', color='red', ha='center', va='center')
    axes[1, 0].set_title("Threshold")

    # plot the weights
    for p, name in enumerate(["right", "left", "up", "down"]):
        state_nump = connections_and_strength[p].numpy()
        axes[2, p].set_xticks(ticks=range(state_nump.shape[1]), labels=range(state_nump.shape[1]))    # Remove axis ticks
        axes[2, p].set_yticks(ticks=range(state_nump.shape[0]), labels=range(state_nump.shape[0]))
        axes[2, p].imshow(state_nump, cmap='gray', aspect='equal', vmin=0, vmax=1)
        for i in range(state_nump.shape[0]): 
            for j in range(state_nump.shape[1]):  axes[2, p].text(j, i, f'{state_nump[i, j]:.1f}', color='red', ha='center', va='center')
        axes[2, p].set_title(name)

    # Hide the unused subplots
    axes[0, 2].axis('off')  # Turn off the 3rd subplot
    axes[0, 3].axis('off')  # Turn off the 3rd subplot
    axes[1, 1].axis('off')  # Turn off the 4th subplot
    axes[1, 2].axis('off')  # Turn off the 4th subplot
    axes[1, 3].axis('off')  # Turn off the 4th subplot
    plt.tight_layout()
    plt.pause(0.000001)





    # def learn_step(self, input, target):
    #     # we want to implement:
    #     # 1. STDP: Spike-timing-dependent plasticity.
    #     # 2. Homeostasis: Keep the average firing rate of the neurons at a certain level.
    #     # 3. Inhibition: Inhibit neurons that are too active.
    #     # 4. Spontaneous firing: Add a small probability of spontaneous firing.
    #     # 5. Learning from input-target error.
    #     # 6. Learning from world model prediction error. 

    #     # hebbian learning:
    #     pass



        # self.state = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        #                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.8],
        #                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        #                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float16)
        # self.threshold = torch.tensor([[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        #                                [1.0, 0.5, 0.3, 0.3, 0.3, 0.3],  # minumum threshold is 1.0!!! this ensures the sumultanious pulsing works!
        #                                [1.0, 1.0, 1.0, 0.3, 1.0, 0.3],
        #                                [1.0, 1.0, 1.0, 0.3, 0.3, 0.3]], dtype=torch.float16)
        # self.connections_and_strength = [torch.tensor([[0.6, 0.6, 0.6, 0.6, 0.6, 0.6],
        #                                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],# right # existance = connection, value = weight 
        #                                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        #                                                [0.0, 0.0, 0.0, 0.4, 0.4, 0.0]], dtype=torch.float16),
        #                             torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        #                                           [0.1, 0.1, 0.1, 0.4, 0.4, 0.4],# left
        #                                           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        #                                           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float16),
        #                             torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        #                                           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],# up
        #                                           [0.0, 0.0, 0.0, 0.0, 0.0, 0.4],
        #                                           [0.0, 0.0, 0.0, 0.0, 0.0, 0.4]], dtype=torch.float16),
        #                             torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        #                                           [0.0, 0.0, 0.0, 0.4, 0.0, 0.0],# down
        #                                           [0.0, 0.0, 0.0, 0.4, 0.0, 0.0],
        #                                           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float16),]
