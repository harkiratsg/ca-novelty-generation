""" Neural Cellular Automata """

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import random

from utils import *


SIZE = 32
TRAIN_GEN_STEPS = 4
TEST_GEN_STEPS = 4
BATCH_SIZE = 1
NUM_EPOCHS = 100


def train(model, start, width, num_epochs, batch_size, gen_steps, loss_func):
    nca.train()
    
    nca_optimizer = optim.Adam(nca.parameters(), lr=0.001)
    losses = []
    
    # Run for specified epochs
    for e in range(num_epochs):
        # Clear gradient
        nca_optimizer.zero_grad()
        
        # Compute loss for one batch
        loss = torch.tensor(0.)
        for b in range(batch_size):
            loss += loss_func(generate_nca_states(nca, torch.rand(WIDTH).detach().clone(), gen_steps))
        
        # Backrop
        loss.backward()
        nca_optimizer.step()
        
        # Save and print loss
        if e % 100 == 0:
            losses.append(loss.item())
            print("{}/{} Loss: ".format(e, num_epochs), loss.item())
    
    return nca, losses

def avg_square_diff(sample):
    sum_diff = 0
    for i in range(sample.shape[0]):
        for j in range(sample.shape[0]):
            sum_diff += torch.mean(torch.pow(sample[i] - sample[j], 2)).item()
    return sum_diff / (sample.shape[0] ** 2)

def sample_sparsity(sample, k):
    sparsities = []
    for i in range(sample.shape[0]):
        sparsities.append(sparsity(sample, s, k))
    return sum(sparsities) / len(sparsities)
        
def evaluate_generated(generated, sample_steps):
    avg_square_diffs = []
    sparsities = []
    entropies = []
    
    for i in range(generated.shape[0] - sample_steps):
        sample = generated[i:i+sample_steps, :]
        
        avg_square_diffs.append(avg_square_diff(sample))
        sparsities.append(sparsity(generated[:i+1], i, 4))
        entropies.append(entropy(sample))
        
    return steps_to_repeat(generated), avg_square_diffs, sparsities, entropies

# Loss functions

def sparsity_cont(state, prev_states, k):
    avg_dist = torch.tensor(0.)
    for prev_state in prev_states:
        d = torch.sum(torch.pow(prev_state - state, 2))
        w = 1 / (1 + torch.exp(2 * (d - k * 2) + 1))
        avg_dist += w * d
    return avg_dist

def avg_sparsity(states, k):
    avg_s = torch.tensor(0.)
    for i in range(1, states.shape[0]):
        avg_s += sparsity_cont(states[i], states[:i], k)
    return avg_s / (states.shape[0] - 1)

def sparsity_loss(states):
    return -avg_sparsity(states, 4)

def spread(state):
    return torch.std(torch.pow(state - torch.mean(state), 2))

def avg_spread_change(states):
    avg_s = torch.tensor(0.)
    for i in range(1, states.shape[0]):
        avg_s += torch.pow(spread(states[i]) - spread(states[i-1]), 2)
    return avg_s / (states.shape[0] - 1)

def spread_change_loss(states):
    return -torch.exp(avg_spread_change(states))
    

if __name__ == "__main__":
    WIDTH = 10
    
    NUM_EPOCHS = 4000
    RES = 2
    INITIAL_POP = 50
    PRESSURE = 0.4
    MUT_RATE = 0.01
    MUT_DEGREE = 0.1
    GEN_STEPS = 100
    
    EVAL_SAMPLES = 2
    EVAL_GEN_STEPS = 100
    
    # Start state
    start = torch.rand(WIDTH)
    
    #g = generate_continuous_rule_states(get_random_continious_rule(3), np.random.rand(10), 100)
    #plt.imshow(g)
    #print(spread_change_loss(torch.tensor(g)))
    #print(dsfds)
    
    # Define NCA model
    nca = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=1, out_channels=80, kernel_size=3, padding=1),
            torch.nn.Sigmoid(),
            torch.nn.Conv1d(in_channels=80, out_channels=400, kernel_size=1, padding=0),
            torch.nn.Sigmoid(),
            torch.nn.Conv1d(in_channels=400, out_channels=80, kernel_size=1, padding=0),
            torch.nn.Sigmoid(),
            torch.nn.Conv1d(in_channels=80, out_channels=1, kernel_size=1, padding=0),
            torch.nn.Sigmoid()
        )
    
    # Train loop
    nca, losses = train(nca, start, WIDTH, NUM_EPOCHS, 2, 10, spread_change_loss)
    
    plt.plot(losses)
    plt.show()
    
    # Evaluate
    with torch.no_grad():
        nca.eval()
        generated_nca = generate_nca_states(nca, start.detach().clone(), 100)
        print(sparsity_loss(generated_nca))
    
    
    plt.imshow(generated_nca)
    plt.show()
