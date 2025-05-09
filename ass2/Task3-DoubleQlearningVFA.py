import numpy as np
from tqdm import tqdm 
import matplotlib.pyplot as plt
import scipy.stats as stats
import math

########################################################################################
##    Double Q-Learning with Linear Function Approximation using One-Hot Encoding     ##
########################################################################################

# Define MDP parameters
actions = (0, 1)  # (0=do nothing, 1=maintenance)
xi = (15, 30, 50) # break threshold per type of component
S = [list(range(x + 1)) for x in xi]
C = tuple([[0, 1]] * x + [[math.inf, 5]] for x in xi)
gamma = 0.9  # Discount factor

# Fix the random seed for reproducibility
np.random.seed(0)

# Utilities
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

def print_double_policies_and_weights(weights1, weights2):
    """
    Print the derived policy (using combined Q-values) and the learned weight vectors for each component type.
    """
    print("Optimal Policy (0 = Do nothing, 1 = Maintenance):")
    for comp_type in range(len(xi)):
        policy = []
        for s in range(xi[comp_type] + 1):
            if s == xi[comp_type]:
                policy.append(1)
            else:
                Q0 = get_Q_value(weights1, comp_type, s, 0) + get_Q_value(weights2, comp_type, s, 0)
                Q1 = get_Q_value(weights1, comp_type, s, 1) + get_Q_value(weights2, comp_type, s, 1)
                policy.append(0 if Q0 >= Q1 else 1)
        print(f"Component Type {comp_type + 1} (Failure Threshold = {xi[comp_type]}):")
        print(policy)
    
    print("\nWeights for each Component Type (Double Q-Learning):")
    for comp_type in range(len(xi)):
        print(f"Component Type {comp_type + 1} (Failure Threshold = {xi[comp_type]}):")
        print("Weights1:", weights1[comp_type])
        print("Weights2:", weights2[comp_type])

# Transition Probabilities
p_zero = 0.5            # 50% probability of zero inflation
dist_name = 'poisson'   # Base distribution
lambda_poisson = 4      # Poisson mean

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

# Feature Extraction and Q-value Calculation
def feature_vector(comp_type, s, a):
    """
    One-hot encoding for (s, a) given the component type.
    The feature vector length is: (xi[comp_type] + 1) * 2.
    Each state-action pair is mapped as: index = s * 2 + a.
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

# e-greedy action determination for Double Q-learning with VFA
def choose_action_double(weights1, weights2, comp_type, s, epsilon):
    """
    Chooses an action using an epsilon-greedy policy based on the combined Q-values from both estimators.
    
    Parameters:
        weights1, weights2: Lists of weight vectors for each Q-estimator.
        comp_type: Integer index for the component type.
        s: Current state (an integer) for the selected component type.
        epsilon: Exploration probability.
        
    Returns:
        a: Chosen action (0 or 1). Returns 1 (maintenance) if s equals the threshold.
    """
    # Force maintenance if state is at threshold.
    if s == xi[comp_type]:
        return 1
    
    if np.random.random() < epsilon:
        return np.random.randint(2)
    else:
        Q0 = get_Q_value(weights1, comp_type, s, 0) + get_Q_value(weights2, comp_type, s, 0)
        Q1 = get_Q_value(weights1, comp_type, s, 1) + get_Q_value(weights2, comp_type, s, 1)
        return 0 if Q0 >= Q1 else 1

# Double Q-Learning with VFA
def run_Double_QLearning_VFA(nEpisodes, lengthEpisode, initial_epsilon, initial_alpha, min_epsilon=0.01, decay_rate=5000, 
                             delta=1e-5, patience=None):
    
    #* Initialize weight vectors as 0 (one for each Q-estimator)
    #* two sets of weight vectors for each component type.
    weights1 = []
    weights2 = []
    for comp_type in range(len(xi)):
        feature_dim = (xi[comp_type] + 1) * 2
        weights1.append(np.zeros(feature_dim))
        weights2.append(np.zeros(feature_dim))
    
    # initialize other things
    TD_errors = []
    prev_policy = None
    stable_count = 0

    #* Repeat (for each episode):
    for i in tqdm(range(nEpisodes), desc="Episodes (Double Q-Learning)"):
        # decay learning rate and epsilon over episodes.
        alpha = initial_alpha / (1 + i / 1000)
        epsilon = max(min_epsilon, initial_epsilon * np.exp(-i / decay_rate))
        
        #* Initialize S
        comp_type = np.random.randint(0, 3)
        s = S[comp_type][0]

        # initialize max TD error
        max_TD_error = 0

        #* Repeat (for each step of episode):
        for _ in range(lengthEpisode):
            #* Choose A from S using  Q using e-greedy pi(s)=argmax_a{Q1(s,a)+Q2(s,a)}
            a = choose_action_double(weights1, weights2, comp_type, s, epsilon)
            
            #* Take action A, observe S', comp_type'
            if a == 0:
                # if doing nothing, degradation increases (or stays the same).
                prob_vector = zero_inflated_prob_vector(p_zero, dist_name, (lambda_poisson,), xi[comp_type] - s)
                increments = np.arange(len(prob_vector))
                increment = np.random.choice(increments, p=prob_vector)
                s_prime = s + increment
                s_prime = min(s_prime, xi[comp_type])
                comp_type_prime = comp_type
            else:
                # if maintenance, reset degradation and choose a new component type.
                s_prime = 0
                comp_type_prime = np.random.randint(0, 3)
            
            #* Take action A, observe R
            r = -C[comp_type][s][a]
            
            #* Double Q-Learning Update: randomly decide which estimator to update:
            #* do Q1(st,at ) <- Q1(st,at) + alpha*(rt + gamma*Q2(s_{t+1}, argmax_a{Q1(st+1, a)}) -  Q1(st, at))
            #* or do Q2(st,at ) <- Q2(st,at) + alpha*(rt + gamma*Q1(s_{t+1}, argmax_a{Q2(st+1, a)}) -  Q2(st, at))
            if np.random.rand() < 0.5:
                # Update weights1:
                Q_sa = get_Q_value(weights1, comp_type, s, a)
                
                # Select next action using weights1.
                if s_prime == xi[comp_type_prime]:
                    a_star = 1  # Force maintenance.
                else:
                    Q1_0 = get_Q_value(weights1, comp_type_prime, s_prime, 0)
                    Q1_1 = get_Q_value(weights1, comp_type_prime, s_prime, 1)
                    a_star = 0 if Q1_0 >= Q1_1 else 1
                
                # Evaluate the next state using weights2.
                Q2_next = get_Q_value(weights2, comp_type_prime, s_prime, a_star)
                target = r + gamma * Q2_next
                TD_error = target - Q_sa
                
                # Update weights1 for the current component type.
                weights1[comp_type] += alpha * TD_error * feature_vector(comp_type, s, a)
            else:
                # Update weights2:
                Q_sa = get_Q_value(weights2, comp_type, s, a)
                
                # Select next action using weights2.
                if s_prime == xi[comp_type_prime]:
                    a_star = 1
                else:
                    Q2_0 = get_Q_value(weights2, comp_type_prime, s_prime, 0)
                    Q2_1 = get_Q_value(weights2, comp_type_prime, s_prime, 1)
                    a_star = 0 if Q2_0 >= Q2_1 else 1
                
                # Evaluate the next state using weights1.
                Q1_next = get_Q_value(weights1, comp_type_prime, s_prime, a_star)
                target = r + gamma * Q1_next
                TD_error = target - Q_sa
                
                # Update weights2 for the current component type.
                weights2[comp_type] += alpha * TD_error * feature_vector(comp_type, s, a)
            
            # update max TD error
            if abs(alpha * TD_error) > max_TD_error:
                max_TD_error = abs(alpha * TD_error)
            
            #* S <- S' (and comp_type <- comp_type')
            s = s_prime
            comp_type = comp_type_prime
        
        TD_errors.append(max_TD_error)
        
        # early stopping based on policy stability.
        if patience is not None:
            current_policy = []
            for comp in range(len(xi)):
                # derive policy for each state of component type 'comp'
                policy = []
                for s_val in range(xi[comp] + 1):
                    if s_val == xi[comp]:
                        policy.append(1)
                    else:
                        Q0 = get_Q_value(weights1, comp, s_val, 0) + get_Q_value(weights2, comp, s_val, 0)
                        Q1 = get_Q_value(weights1, comp, s_val, 1) + get_Q_value(weights2, comp, s_val, 1)
                        policy.append(0 if Q0 >= Q1 else 1)
                current_policy.append(np.array(policy))
            
            # check if the policy is stable
            if prev_policy is not None and all(np.array_equal(cp, pp) for cp, pp in zip(current_policy, prev_policy)):
                stable_count += 1
                if stable_count >= patience:
                    print(f"Stopped early at episode {i+1} — policy stable for {patience} episodes.")
                    break
            else:
                stable_count = 0
                prev_policy = current_policy
        
        # Early stopping based on TD error convergence.
        if max_TD_error < delta:
            return TD_errors, (weights1, weights2)
    
    return TD_errors, (weights1, weights2)

# learning parameters.
nEpisodes = int(1e4)
lengthEpisode = int(1e3)
initial_epsilon = 0.2
initial_alpha = 0.05
decay_rate = 10_000

td_errors_double, (weights1, weights2) = run_Double_QLearning_VFA(
    nEpisodes, lengthEpisode, initial_epsilon, initial_alpha, decay_rate=decay_rate, patience=200
)

final_results_double = {f"Double Q-Learning: $\\epsilon$={initial_epsilon}, $\\alpha$={initial_alpha}": {"td_errors": td_errors_double}}
plot_td_errors(final_results_double, title="Double Q-Learning TD Error Convergence")
print_double_policies_and_weights(weights1, weights2)