import numpy as np
import matplotlib.pyplot as plt
import torch
import math
import os

def evaluate(rules, path, width, eval_samples, eval_gen_steps):
    print("\nEvaluation")
    eval_repeat_steps = []
    eval_total_sparsity = []
    eval_total_entropy_change = []
    
    if not os.path.exists(path):
        os.makedirs(path)
    
    for r, rule in enumerate(rules):
        for i in range(eval_samples):
            generated = generate_continuous_rule_states(rule, np.random.rand(width), eval_gen_steps)
            eval_repeat_steps.append(steps_to_repeat(generated))
            eval_total_sparsity.append(total_sparsity(generated, 4))
            eval_total_entropy_change.append(total_entropy_change(generated, 5))
            
            plt.imsave(f'{path}generated_{r}_{i}.png', generated)
    
    print(f"Steps to Repeat: Mean:{np.mean(np.array(eval_repeat_steps))} Standard Deviation: {np.std(np.array(eval_repeat_steps))}")
    print(f"Total Sparsity : Mean:{np.mean(np.array(eval_total_sparsity))} Standard Deviation: {np.std(np.array(eval_total_sparsity))}")
    print(f"Total E-Change : Mean:{np.mean(np.array(eval_total_entropy_change))} Standard Deviation: {np.std(np.array(eval_total_entropy_change))}")

def sparsity(state, prev_states, k):
    k = min([prev_states.shape[0], k])
    
    dists = []
    for prev_state in prev_states:
        dists.append((np.sum(np.power((prev_state - state), 2))) ** 0.5)
    dists.sort()
    
    return sum(dists[:k]) / k

def total_sparsity(states, k):
    sparsity_values = []
    for i in range(1, states.shape[0]):
        sparsity_values.append(sparsity(states[i], states[:i], k))
    return sum(sparsity_values)

def entropy(states, g):
    discretized_state = np.floor(states.flatten() * g * 0.9999).tolist()
    e = 0
    for i in range(g):
        p = discretized_state.count(i) / len(discretized_state)
        e -= p * math.log(p + 0.0001) / math.log(g)
    return e / states.shape[0]

def total_entropy_change(states, g):
    ec = 0
    for i in range(1, states.shape[0]):
        ec += abs(entropy(states[i-1], g) - entropy(states[i], g))
    return ec

# Plot states
def plot_states(states):
    plt.imshow(states)
    plt.show()
    
def plot_evaluation(t, avg_sparsities_diff, sparsities, entropies):
    plt.ylim(0, 1)
    plt.xlim(0, len(avg_sparsities_diff))
    plt.axvspan(t-1, len(avg_sparsities_diff), facecolor='0.2', alpha=0.2)
    plt.plot(avg_sparsities_diff)
    plt.plot(sparsities)
    plt.plot(entropies)
    plt.show()

def bin_list_to_dec(l):
    return int("".join(str(i) for i in l),2)

def steps_to_repeat(states):
    for i in range(1, states.shape[0]):
        if states[i].tolist() in states[:i, :].tolist():
            return i
    return states.shape[0]

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


""" Generate With Rule """

def get_random_continious_rule(resolution):
    return np.random.rand(resolution, resolution, resolution)

# Generate specified rule next state
def generate_discrete_rule_next_state(rule, current_state):
    next_state = torch.zeros(current_state.shape[0])
    current_state = torch.nn.functional.pad(current_state, (1,1), 'constant', 0)
    for i in range(1, current_state.shape[0]-1):
        next_state[i-1] = rule[bin_list_to_dec(current_state[i-1:i+2].int().tolist())]
    return next_state

# Generate n states of rule 26
def generate_discrete_rule_states(rule, start, n):
    states = torch.unsqueeze(start, dim=0)
    for i in range(n):
        next_state = generate_discrete_rule_next_state(rule, states[-1])
        states = torch.cat([states, torch.unsqueeze(next_state, dim=0)], dim=0)
    return states


# Map cell values to single value (Only works for 3D right now)
def map_window_to_values(values, rule):
    mapped_values = np.floor(0.999 * values * rule.shape[0])
    return rule[int(mapped_values[0])][int(mapped_values[1])][int(mapped_values[2])]

# Generate specified rule next state
def generate_continuous_rule_next_state(rule, current_state):
    next_state = np.zeros(current_state.shape[0])
    current_state = np.pad(current_state, (1,), 'constant', constant_values=0)
    for i in range(1, current_state.shape[0]-1):
        next_state[i-1] = map_window_to_values(current_state[i-1:i+2], rule)
    return next_state

# Generate n states of rule 26
def generate_continuous_rule_states(rule, start, n):
    states = np.expand_dims(start, axis=0)
    for i in range(n):
        next_state = generate_continuous_rule_next_state(rule, states[-1])
        states = np.concatenate([states, np.expand_dims(next_state, axis=0)], axis=0)
    return states




""" Generate With NCA """

# Generate states using neural net CA rules
def generate_nca_next_state(model, current_state):
    # Start states with a random vector
    states = torch.unsqueeze(current_state, dim=0)
    
    # Generate next state from rules model given the past state
    return model(torch.unsqueeze(torch.unsqueeze(states[-1], dim=0), dim=0))[0][0]

# Generate states using neural net CA rules
def generate_nca_states(model, start, n):
    # Start states with a random vector
    states = torch.unsqueeze(start, dim=0)
    
    for i in range(n):
        # Generate next state from rules model given the past state
        next_state = model(torch.unsqueeze(torch.unsqueeze(states[-1], dim=0), dim=0))[0][0]
        
        # Add state to states
        states = torch.cat([states, torch.unsqueeze(next_state, dim=0)], dim=0)
    return states
