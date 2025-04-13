import numpy as np
from tqdm import tqdm 
import matplotlib.pyplot as plt
import scipy.stats as stats
import math

###########################
##  Q-Learning with Linear Function Approximation using One-Hot Encoding (Version 1)  ##
###########################

# MDP parameters:
actions = (0, 1)  # 0 = do nothing, 1 = maintenance
xi = (15, 30, 50) # failure thresholds for each component type
S = [list(range(x + 1)) for x in xi]
# Cost structure: for each component type, the cost array is defined for each state and action.
# (Here we assume that for states less than the threshold, costs are defined by [0, 1] and at the threshold,
# maintenance is forced with cost 5, and doing nothing is infeasible (set to infinity).)
C = tuple([[0, 1]] * x + [[math.inf, 5]] for x in xi)
gamma = 0.9  # Discount factor

# Fix the random seed for reproducibility
np.random.seed(0)

# -----------------------
# Version 1 Feature Functions
# -----------------------

def feature_vector(comp_type, s, a):
    """
    One-hot encoding for (s, a) given the component type.
    The feature vector length is: (xi[comp_type] + 1) * 2.
    Each state–action pair is mapped as:
         index = s * 2 + a.
    """
    length = (xi[comp_type] + 1) * 2
    phi = np.zeros(length)
    idx = s * 2 + a
    phi[idx] = 1.0
    return phi

def get_Q_value(weights, comp_type, s, a):
    """
    Returns the approximated Q-value as the dot product between the weight vector
    (for the given component type) and the one-hot feature vector for (s, a).
    """
    phi = feature_vector(comp_type, s, a)
    return np.dot(weights[comp_type], phi)

def choose_action(weights, comp_type, s, epsilon):
    """
    Epsilon-greedy action selection.
    Force maintenance (action 1) when s reaches the threshold.
    """
    if s == xi[comp_type]:
        return 1  # Force maintenance at the threshold.
    if np.random.random() < epsilon:
        return np.random.randint(2)
    else:
        Q0 = get_Q_value(weights, comp_type, s, 0)
        Q1 = get_Q_value(weights, comp_type, s, 1)
        return 0 if Q0 >= Q1 else 1

def zero_inflated_prob_vector(p_zero, dist_name, dist_params, s):
    """
    Computes a probability vector for a zero-inflated random variable.
    """
    base_dist = getattr(stats, dist_name)
    if s == 0:
        prob_vector = [p_zero]
    else:
        pmf_values = (1 - p_zero) * base_dist.pmf(np.arange(s), *dist_params)
        pmf_values[0] += p_zero
        p_s_or_more = 1 - np.sum(pmf_values)
        prob_vector = np.append(pmf_values, p_s_or_more)
    return prob_vector

def plot_td_errors(results, title="TD Error Convergence"):
    """
    Utility function to plot the maximum TD error per episode.
    """
    plt.figure(figsize=(12, 7))
    for label, data in results.items():
        plt.plot(data["td_errors"], label=label)
    plt.xlabel("Episode")
    plt.ylabel("Max TD Error")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def print_policies_and_weights(weights):
    """
    Print the derived policy (action chosen for each state) and the learned weight vectors.
    """
    print("Optimal Policy (0 = Do nothing, 1 = Maintenance):")
    for comp_type in range(len(xi)):
        policy = []
        for s in range(xi[comp_type] + 1):
            if s == xi[comp_type]:
                policy.append(1)
            else:
                Q0 = get_Q_value(weights, comp_type, s, 0)
                Q1 = get_Q_value(weights, comp_type, s, 1)
                policy.append(0 if Q0 >= Q1 else 1)
        print(f"Component Type {comp_type + 1} (Failure Threshold = {xi[comp_type]}):")
        print(policy)
    
    print("\nWeights for each Component Type:")
    for comp_type in range(len(xi)):
        print(f"Component Type {comp_type + 1} (Failure Threshold = {xi[comp_type]}):")
        print(weights[comp_type])

def run_QLearning(nEpisodes, lengthEpisode, initial_epsilon, initial_alpha, min_epsilon=0.01, decay_rate=5000, 
                   delta=1e-5, patience=None):
    # Initialize a weight vector per component type.
    # Each weight vector has dimension = (xi[comp_type] + 1) * 2.
    weights = []
    for comp_type in range(len(xi)):
        feature_dim = (xi[comp_type] + 1) * 2
        weights.append(np.zeros(feature_dim))
    # weights = tuple(weights)
    
    TD_errors = []
    prev_policy = None
    stable_count = 0

    # Parameters for the zero-inflated process
    p_zero = 0.5      # 50% probability of zero inflation
    dist_name = 'poisson'
    lambda_poisson = 4

    for i in tqdm(range(nEpisodes), desc="Episodes"):
        # Decay learning rate and epsilon over episodes.
        alpha = initial_alpha / (1 + i / 1000)
        epsilon = max(min_epsilon, initial_epsilon * np.exp(-i / decay_rate))
        
        # Initialize state: select a random component type, starting at s = 0.
        comp_type = np.random.randint(0, 3)
        s = S[comp_type][0]
        max_TD_error = 0

        for _ in range(lengthEpisode):
            # Select an action using the epsilon-greedy policy.
            a = choose_action(weights, comp_type, s, epsilon)
            
            # Execute action a:
            if a == 0:
                # If doing nothing, the degradation increases.
                prob_vector = zero_inflated_prob_vector(p_zero, dist_name, (lambda_poisson,), xi[comp_type] - s)
                increments = np.arange(len(prob_vector))
                increment = np.random.choice(increments, p=prob_vector)
                s_prime = s + increment
                s_prime = min(s_prime, xi[comp_type])  # Ensure s' does not exceed the threshold.
                comp_type_prime = comp_type
            else:
                # If maintenance is performed, degradation resets and a new component type is chosen.
                s_prime = 0
                comp_type_prime = np.random.randint(0, 3)
            
            # Observe the reward.
            r = -C[comp_type][s][a]
            
            # Compute current Q-value and target value.
            Q_sa = get_Q_value(weights, comp_type, s, a)
            if s_prime == xi[comp_type_prime]:
                maxNextQ = get_Q_value(weights, comp_type_prime, s_prime, 1)
            else:
                Q0 = get_Q_value(weights, comp_type_prime, s_prime, 0)
                Q1 = get_Q_value(weights, comp_type_prime, s_prime, 1)
                maxNextQ = max(Q0, Q1)
            
            TD_error = r + gamma * maxNextQ - Q_sa
            phi = feature_vector(comp_type, s, a)
            # Update the weight vector for the current component type.
            weights[comp_type] += alpha * TD_error * phi
            
            if abs(alpha * TD_error) > max_TD_error:
                max_TD_error = abs(alpha * TD_error)
            
            s = s_prime
            comp_type = comp_type_prime
        
        TD_errors.append(max_TD_error)
        
        # Early stopping based on policy stability.
        if patience is not None:
            current_policy = []
            for comp in range(len(weights)):
                policy = []
                for s_val in range(xi[comp] + 1):
                    if s_val == xi[comp]:
                        policy.append(1)
                    else:
                        Q0 = get_Q_value(weights, comp, s_val, 0)
                        Q1 = get_Q_value(weights, comp, s_val, 1)
                        policy.append(0 if Q0 >= Q1 else 1)
                current_policy.append(np.array(policy))
            if prev_policy is not None and all(np.array_equal(cp, pp) for cp, pp in zip(current_policy, prev_policy)):
                stable_count += 1
                if stable_count >= patience:
                    print(f"Stopped early at episode {i+1} — policy stable for {patience} episodes.")
                    break
            else:
                stable_count = 0
                prev_policy = current_policy
        
        if max_TD_error < delta:
            return TD_errors, weights
    
    return TD_errors, weights

# Learning parameters.
nEpisodes = pow(10,4)
lengthEpisode = pow(10,3)
initial_epsilon = 0.2
initial_alpha = 0.05
decay_rate = 10_000

td_errors, weights = run_QLearning(nEpisodes, lengthEpisode, initial_epsilon, initial_alpha, decay_rate=decay_rate, patience=200)
final_results = {f"$\\epsilon$={initial_epsilon}, $\\alpha$={initial_alpha}": {"td_errors": td_errors}}
plot_td_errors(final_results, title="Final TD Error Convergence")
print_policies_and_weights(weights)