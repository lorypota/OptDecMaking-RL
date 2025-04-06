import numpy as np
import math
import scipy.stats as stats
from tqdm import tqdm 
import matplotlib.pyplot as plt
import random

# Utility
np.random.seed(0)
random.seed(0)

def plot_convergence(rho_histories, g_values, filename='rho_convergence_by_component.png'):
    """
    Plot the convergence of average cost estimates by component type.
    
    Parameters:
    - rho_histories: List of histories of rho values for each component
    - g_values: List of reference g values for each component
    - filename: Name of file to save the plot to
    """
    plt.figure(figsize=(10, 6))
    
    component_lines = []
    for comp_type in range(len(rho_histories)):
        line, = plt.plot(range(0, len(rho_histories[comp_type])*10, 10), 
                         rho_histories[comp_type], 
                         label=f'Component Type {comp_type+1}')
        component_lines.append(line)
    
    # Reference values
    for comp_type in range(len(g_values)):
        plt.axhline(y=g_values[comp_type], 
                    color=component_lines[comp_type].get_color(), 
                    linestyle='--', 
                    label=f'Model g{comp_type+1} = {g_values[comp_type]:.4f}')
    
    plt.xlabel('Episode')
    plt.ylabel('Average Cost Estimate (ρ)')
    plt.title('Convergence of Average Cost Estimates by Component Type')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.show()

# Define MDP parameters
xi = (15, 30, 50)
C = tuple([[0, 1]] * x + [[math.inf, 5]] for x in xi)

# Transition probability parameters
p_zero = 0.5  
dist_name = 'poisson'
lambda_poisson = 4

def zero_inflated_prob_vector(p_zero, dist_name, dist_params, s):
    """
    Computes a probability vector for a zero-inflated random variable.

    Parameters:
    - p_zero (float): Probability of zero inflation.
    - dist_name (str): Name of the base distribution ('poisson', 'nbinom', 'binom').
    - dist_params (tuple): Parameters of the base distribution.
    - s (int): Threshold for "s or greater" category.

    Returns:
    - np.array: Probability vector of length (s+1) where:
        - First element: P(X=0)
        - Second element: P(X=1)
        - ...
        - Second-to-last element: P(X=s-1)
        - Last element: P(X >= s)=1-(P(X=0)+...+P(X=s-1))
    """
    # Get the chosen probability mass function (PMF)
    base_dist = getattr(stats, dist_name)

    if s==0:
        prob_vector = [p_zero]
    else:
        # Compute probabilities for values 0 to (s-1)
        pmf_values = (1 - p_zero) * base_dist.pmf(np.arange(s), *dist_params)
        
        # Adjust probability of zero (includes zero-inflation)
        pmf_values[0] += p_zero
        
        # Compute probability for X ≥ s
        p_s_or_more = 1 - np.sum(pmf_values)
        
        # Append P(X >= s) as the last element
        prob_vector = np.append(pmf_values, p_s_or_more)
    
    return prob_vector

def build_states():
    """Build the state space"""
    states = []
    for comp_type in range(len(xi)):
        for s in range(xi[comp_type] + 1):
            states.append((comp_type, s))
    return states

def sample_transition(state, action):
    """
    Sample a transition (next_state, cost) given current state and action.
    """
    comp_type, s = state

    cost = C[comp_type][s][action]
    
    # Handle special case
    if s == xi[comp_type] and action == 0:
        return None, cost
    
    # Sample next state
    if action == 0:  # Do nothing -> sample increment from zero-inflated Poisson
        delta_max = xi[comp_type] - s
        prob_vector = zero_inflated_prob_vector(p_zero, dist_name, (lambda_poisson,), delta_max)
        increment = np.random.choice(range(len(prob_vector)), p=prob_vector)
        next_s = min(s + increment, xi[comp_type])
        next_state = (comp_type, next_s)
    else:  # Maintenance -> reset to 0
        next_state = (comp_type, 0)
    
    return next_state, cost

def r_learning(max_episodes=10000, component_params=None):
    """
    R-Learning algorithm for average cost criteria
    
    Parameters:
    - max_episodes: Maximum number of episodes to run
    - component_params: Dictionary of component-specific parameters
    
    Returns:
    - policies, rhos, rho_histories
    """
    actions = [0, 1]
    num_components = len(xi)
    
    # Initialize arrays instead of dictionaries
    policies = [None] * num_components
    rhos = [0.0] * num_components
    rho_histories = [[] for _ in range(num_components)]
    
    for comp_type in range(num_components):
        params = component_params[comp_type]
        
        Q = {}
        for s in range(xi[comp_type] + 1):
            Q[(comp_type, s)] = {0: 0.0, 1: 0.0}
        
        rho = 0.0
        rho_history = []
        
        for episode in tqdm(range(max_episodes), desc=f"R-Learning component {comp_type+1}"):
            epsilon = max(params['epsilon_start'] * (params['epsilon_decay'] ** episode), 0.1)
            
            # Start at random state for this component
            s = random.randint(0, xi[comp_type])
            state = (comp_type, s)
            
            for step in range(params['max_steps_per_episode']):
                # Choose action using epsilon-greedy
                if random.random() < epsilon:
                    # Explore
                    action = random.choice(actions)
                else:
                    # Exploit
                    comp_type, s = state
                    if s == xi[comp_type]:
                        action = 1
                    else:
                        if Q[state][0] <= Q[state][1]:
                            action = 0
                        else:
                            action = 1

                next_state, cost = sample_transition(state, action)
                
                # If invalid transition, force maintenance
                if next_state is None:
                    action = 1
                    next_state, cost = sample_transition(state, action)
                
                # Get minimum Q-values
                min_q_next = min(Q[next_state][0], Q[next_state][1])
                min_q_current = min(Q[state][0], Q[state][1])
                
                Q[state][action] += params['alpha'] * (cost - rho + min_q_next - Q[state][action])
                rho += params['beta'] * (cost + min_q_next - min_q_current - rho)
                
                if episode % 10 == 0 and step == 0:
                    rho_history.append(rho)
                
                state = next_state
                s = state[1]
            
            # Extract current policy
            current_policy = [0] * (xi[comp_type] + 1)
            for s in range(xi[comp_type] + 1):
                state = (comp_type, s)
                if s == xi[comp_type]:
                    current_policy[s] = 1
                else:
                    current_policy[s] = 0 if Q[state][0] <= Q[state][1] else 1
            
            # Check for convergence
            if episode > 1000 and len(rho_history) >= params['conv_window']+1:
                if abs(rho_history[-1] - rho_history[-params['conv_window']]) < params['conv_threshold']:
                    print(f"Component {comp_type+1} converged after {episode} episodes")
                    break
    
        # Use the final policy
        policy = [0] * (xi[comp_type] + 1)
        for s in range(xi[comp_type] + 1):
            state = (comp_type, s)
            # Force maintenance at threshold
            if s == xi[comp_type]:
                policy[s] = 1
            else:
                if Q[state][0] <= Q[state][1]:
                    policy[s] = 0
                else:
                    policy[s] = 1
        
        policies[comp_type] = policy
        rhos[comp_type] = rho
        rho_histories[comp_type] = rho_history
    
    # Return results
    return policies, rhos, rho_histories

def build_model_based_policy():
    """
    Build the policy from the model-based solution (from RVI algorithm results)
    """
    policy = {}
    
    # Type 1
    type1_decisions = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
    for s, a in enumerate(type1_decisions):
        policy[(0, s)] = a
        
    # Type 2
    type2_decisions = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    for s, a in enumerate(type2_decisions):
        policy[(1, s)] = a
        
    # Type 3
    type3_decisions = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    for s, a in enumerate(type3_decisions):
        policy[(2, s)] = a
        
    return policy

# Results from model-based solution (from previous run with RVI)
g_1 = 0.1808
g_2 = 0.0795
g_3 = 0.0449
model_policy = build_model_based_policy()

# Define component-specific parameters based on state space size
component_params = {
    0: {
        'alpha': 0.01,
        'beta': 0.01,
        'epsilon_decay': 0.999,
        'epsilon_start': 1.0,
        'max_steps_per_episode': 75,
        'conv_window': 100,
        'conv_threshold': 1e-5
    },
    1: {
        'alpha': 0.01,
        'beta': 0.01,
        'epsilon_decay': 0.999,
        'epsilon_start': 1.0,
        'max_steps_per_episode': 150,
        'conv_window': 100,
        'conv_threshold': 1e-5
    },
    2: {
        'alpha': 0.01,
        'beta': 0.01,
        'epsilon_decay': 0.999,
        'epsilon_start': 1.0,
        'max_steps_per_episode': 250,
        'conv_window': 100,
        'conv_threshold': 1e-5
    }
}


states = build_states()
policies, rhos, rho_histories = r_learning(
    max_episodes=10000,
    component_params=component_params,
)

print("\nOptimal policies from component-specific R-Learning:")
for comp_type in range(len(xi)):
    print(f"Component Type {comp_type+1} (xi={xi[comp_type]}):")
    print(f"Average Cost: {rhos[comp_type]:.4f} (Model: {[g_1, g_2, g_3][comp_type]:.4f})")
    print(f"Policy: {policies[comp_type]}")
    print()

plot_convergence(rho_histories, [g_1, g_2, g_3])