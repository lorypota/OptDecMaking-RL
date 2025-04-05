import numpy as np
import math
import scipy.stats as stats
from tqdm import tqdm 
import matplotlib.pyplot as plt
import random

# Utility
np.random.seed(0)
random.seed(0)

def print_optimal_policy(policy, xi):
    """Utility function to print policy per component type"""
    policy_by_comp = {}
    for (comp_type, s), a in policy.items():
        if comp_type not in policy_by_comp:
            policy_by_comp[comp_type] = {}
        policy_by_comp[comp_type][s] = a
        
    for comp_type in sorted(policy_by_comp.keys()):
        decisions = [policy_by_comp[comp_type].get(s, None) for s in range(xi[comp_type]+1)]
        print(f"Component Type {comp_type+1} (Failure Threshold = {xi[comp_type]}):")
        print("Action:", decisions)
        print()

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
    Sample a transition (next_state, cost) given current state and action
    without using the full model knowledge.
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
    else:  # Maintenance -> reset to 0 and randomly select component type
        new_comp = np.random.randint(0, len(xi))
        next_state = (new_comp, 0)
    
    return next_state, cost

def r_learning(states, max_episodes=10000, max_steps_per_episode=100, 
               alpha=0.1, beta=0.01, epsilon_start=1.0, epsilon_decay=0.999):
    """
    R-Learning algorithm for average cost reinforcement learning
    
    Parameters:
    - states: List of all possible states
    - max_episodes: Maximum number of episodes to run
    - max_steps_per_episode: Maximum steps per episode
    - alpha: Learning rate for Q-values
    - beta: Learning rate for average cost (rho)
    - epsilon_start: Initial exploration rate
    - epsilon_decay: Decay rate for exploration
    
    Returns:
    - policy: The learned policy
    - Q: The learned Q-values
    - rho: The estimated average cost
    - rho_history: History of rho values for convergence analysis
    """
    actions = [0, 1]
    Q = {state: {a: 0.0 for a in actions} for state in states}
    rho = 0.0
    
    rho_history = []
    policy_history = []
    
    for episode in tqdm(range(max_episodes), desc="R-Learning"):
        epsilon = max(epsilon_start * (epsilon_decay ** episode), 0.1)
        beta_t = beta * (1 / (1 + 0.0001 * episode))
        
        # Select random initial state (more uniform than starting at (0, T))
        # By selecting state spaces (on T) uniformly at random we sample more frequently from states spaces with a
        # higher number of states (T=3 with 50 states is sampled more than half compared to both T=2 and T=1)
        # This is by design since components with more states generally require more training to learn optimal policies
        state = random.choice(states)
        
        for step in range(max_steps_per_episode):
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
                    action = min(actions, key=lambda a: Q[state][a])

            next_state, cost = sample_transition(state, action)
            
            # If invalid transition, force maintenance
            if next_state is None:
                action = 1
                next_state, cost = sample_transition(state, action)
            
            min_q_next = min(Q[next_state].values())
            min_q_current = min(Q[state].values())
            
            Q[state][action] += alpha * (cost - rho + min_q_next - Q[state][action])
            rho += beta_t * (cost + min_q_next - min_q_current - rho)
            
            if episode % 10 == 0 and step == 0:
                rho_history.append(rho)
            
            state = next_state
        
        # Evaluate policy periodically
        if episode % 100 == 0:
            # Extract current greedy policy
            current_policy = {}
            for s in states:
                comp_type, state_s = s
                # Force maintenance at threshold
                if state_s == xi[comp_type]:
                    current_policy[s] = 1
                else:
                    current_policy[s] = min(actions, key=lambda a: Q[s][a])
            
            policy_history.append(current_policy)
            
            # Check for policy stability (convergence)
            if len(policy_history) >= 5:
                stable = True
                ref_policy = policy_history[-1]
                for prev_policy in policy_history[-5:-1]:
                    if not all(ref_policy[s] == prev_policy[s] for s in states):
                        stable = False
                        break
                
                if stable and episode > max_episodes // 2:
                    print(f"Policy converged after {episode} episodes")
                    break
    
    # Extract final policy
    final_policy = {}
    for s in states:
        comp_type, state_s = s
        # Force maintenance at threshold
        if state_s == xi[comp_type]:
            final_policy[s] = 1
        else:
            final_policy[s] = min(actions, key=lambda a: Q[s][a])
    
    return final_policy, Q, rho, rho_history

def build_model_based_policy():
    """
    Build the policy from the model-based solution (from RVI algorithm results)
    """
    policy = {}
    
    # Type 1
    type1_decisions = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
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
g_model = 0.0747  # Average cost from RVI
model_policy = build_model_based_policy()

# Run R-Learning with selected hyperparameters
states = build_states()
policy_r, Q_r, rho_r, rho_history = r_learning(
    states, 
    max_episodes=50000,
    max_steps_per_episode=1000,
    alpha=0.05,  # Learning rate for Q-values
    beta=0.01,   # Learning rate for average cost estimate
    epsilon_start=1.0,
    epsilon_decay=0.9995
)

print("\nTask 4: Average Cost Reinforcement Learning")
print("\nPart 1: Model-based solution results (from Relative Value Iteration)")
print(f"Long-run average cost: {g_model:.4f}")
print("Optimal policy from model-based solution:")
print_optimal_policy(model_policy, xi)

print("\nPart 2: Model-free solution (R-Learning)")
print(f"Estimated average cost: {rho_r:.4f}")
print("\nOptimal policy from R-Learning:")
print_optimal_policy(policy_r, xi)

# Compare policies
agreements = sum(1 for s in states if model_policy.get(s) == policy_r.get(s))
agreement = agreements / len(states) * 100
print(f"\nPolicy Comparison:")
print(f"Agreement: {agreement:.2f}% ({agreements}/{len(states)} states)")
print(f"Average cost difference: {abs(g_model - rho_r):.4f}")

# Plot convergence of average cost estimate
plt.figure(figsize=(10, 6))
plt.plot(range(0, len(rho_history)*10, 10), rho_history)
plt.axhline(y=g_model, color='r', linestyle='--', label=f'Model-based g = {g_model:.4f}')
plt.xlabel('Episode')
plt.ylabel('Average Cost Estimate (ρ)')
plt.title('Convergence of Average Cost Estimate in R-Learning')
plt.legend()
plt.grid(True)
plt.savefig('rho_convergence.png')
plt.show()