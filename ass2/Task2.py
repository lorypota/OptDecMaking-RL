import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.stats as stats
from tqdm import tqdm 

############################################################
##       Monte Carlo with Exponential Epsilon Decay       ##
############################################################

# Utility
np.random.seed(0)

def plot_episode_returns(results, title="Episode Return Convergence"):
    """
    Utility function to plot episode returns (can be multiple).
    """
    plt.figure(figsize=(12, 7))
    for label, data in results.items():
        plt.plot(data["episode_returns"], label=label)

    plt.xlabel("Episode")
    plt.ylabel("Episode Return")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def print_policies_and_q_values(Q, label=None):
    """
    Utility function to print policies and corresponding Q values (can be multiple)
    """
    if label:
        print(f"\n===== Optimal Policy for {label} =====")
    print("Optimal Policy (0 = Do nothing, 1 = Maintenance):")
    for comp_type in range(len(xi)):
        policy = np.argmax(Q[comp_type], axis=0)
        policy[xi[comp_type]] = 1  # Force maintenance at threshold
        print(f"Component Type {comp_type + 1} (Failure Threshold = {xi[comp_type]}):")
        print(policy)

    print("\nQ-values for each Component Type:")
    for comp_type in range(len(xi)):
        print(f"\nComponent Type {comp_type + 1} (Failure Threshold = {xi[comp_type]}):")
        print(Q[comp_type])

# Define MDP parameters
actions = (0, 1)  # (0=do nothing, 1=maintenance)
xi = (15, 30, 50) # break threshold per type of component
S = [list(range(x + 1)) for x in xi]
C = tuple([[0, 1]] * x + [[math.inf, 5]] for x in xi)
gamma = 0.9  # Discount factor

# Transition probabilities
p_zero = 0.5  # 50% probability of zero inflation
dist_name = 'poisson'  # Base distribution
lambda_poisson = 4  # Poisson mean

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

def choose_action(Q: tuple, comp_type: int, s: int, xi: tuple, epsilon: float) -> int:
    """
    Choose an action using an epsilon-greedy policy with a forced action at the threshold.

    Parameters:
     - Q         : Tuple of numpy arrays representing the Q-table for each component type.
     - comp_type : Integer index indicating the component type.
     - s         : Current state (an integer) for the selected component type.
     - xi        : Tuple containing the failure threshold for each component type.
     - epsilon   : Exploration probability.

    Returns:
     - a         : The chosen action (0 or 1). If s equals the threshold (xi[comp_type]), then returns 1.
    """
    # If state equals the failure threshold, force the maintenance action.
    if s == xi[comp_type]:
        return 1

    # generate random float in the half-open interval [0.0, 1.0)
    if np.random.random() < epsilon:
        # if less than epsilon, choose a random action (exploration)
        return np.random.randint(2)
    else:
        # if greater than epsilon, choose the best action (exploitation)
        return np.argmax(Q[comp_type][:, s])


def run_simulation(nEpisodes, lengthEpisode, initial_epsilon, min_epsilon=0.01, decay_rate=5000,
                   patience=None):
    
    # ----- initialization -----

    #* initialize Q(s,a) arbitrarily for all s in S and a in A_s
    Q = tuple(np.zeros((2, x + 1)) for x in xi)
    #* Keep track of the number of visits N(s,a)
    N = tuple(np.zeros((2, x + 1)) for x in xi)

    prev_policy = None
    stable_count = 0

    episode_returns = []
    
    for episode_nr in tqdm(range(nEpisodes), desc="Episodes"):
        
        # decay epsilon
        epsilon = max(min_epsilon, initial_epsilon * np.exp(-episode_nr / decay_rate))

        # ----- Generate Episode -----       
        
        # initialize S
        comp_type = np.random.randint(0, 3)
        s = 0 #S[comp_type][0]

        episode_transitions = []  # List to store transitions of the current episode

        # for each step:
        for step in range(lengthEpisode):
            # choose action A using e-greedy policy w.r.t. current Q
            a = choose_action(Q, comp_type, s, xi, epsilon)

            # take action A, observe S'
            if a == 0:
                prob_vector = zero_inflated_prob_vector(p_zero, dist_name, (lambda_poisson,), xi[comp_type]-s)
                increments = np.arange(len(prob_vector))
                increment = np.random.choice(increments, p=prob_vector)
                s_prime = s + increment
                if s_prime > xi[comp_type]:
                    s_prime = xi[comp_type]
                comp_type_prime = comp_type
            else:
                s_prime = 0
                comp_type_prime = np.random.randint(0, 3)

            # observe R
            r = -C[comp_type][s][a]

            # append (S, A, R) to a list of episode transitions
            episode_transitions.append((comp_type, s, a, r))

            # transition to S'
            s = s_prime
            comp_type = comp_type_prime

        # ----- Compute returns and update Q -----
        # precompute all returns G_t for the episode in a backward pass
        T = len(episode_transitions) # = lengthEpisode in this case bc no terminal state
        G_t = np.zeros(T)
        G = 0.0
        for t in reversed(range(T)):
            # unpack the transition tuple
            _, _, _, r = episode_transitions[t]

            # recursive computation of G_t
            G = r + gamma*G
            G_t[t] = G

        episode_returns.append(G_t[0])

        # every-visit Monte Carlo: for each state S_t and action A_t in the episode, update Q:
        for t,step in enumerate(episode_transitions):
            # unpack the transition tuple
            comp_type, s, a, _ = step

            # update the visit count: N(s_t, a_t) <- N(s_t, a_t) + 1
            N[comp_type][a,s] += 1

            # update the Q-value: Q(s_t, a_t) <- Q(s_t, a_t) + (1/N(s_t, a_t)) * (G_t - Q(s_t, a_t))
            Q[comp_type][a,s] += (1/N[comp_type][a,s]) * (G_t[t] - Q[comp_type][a,s])


        # ----- Early-stopping based on policy stability -----
        
        if patience is not None:
            current_policy = tuple(np.argmax(Q[comp], axis=0) for comp in range(len(Q)))
            
            # Force maintenance at the threshold
            for k, policy in enumerate(current_policy):
                policy[xi[k]] = 1  

            # Check if the policy is stable
            if prev_policy is not None and all(np.array_equal(cp, pp) for cp, pp in zip(current_policy, prev_policy)):
                stable_count += 1
                if stable_count >= patience:
                    print(f"Stopped early at episode {episode_nr+1} — policy stable for {patience} episodes.")
                    return Q, episode_returns
            else:
                stable_count = 0
                prev_policy = current_policy
        
    return Q, episode_returns

# Final tuned run
nEpisodes = pow(10, 4)
lengthEpisode = pow(10, 3)
initial_epsilon= 0.3
min_epsilon = 0.01
decay_rate = 5_000

Q, episode_returns = run_simulation(nEpisodes, lengthEpisode, initial_epsilon=initial_epsilon, min_epsilon=min_epsilon, decay_rate=decay_rate, patience=100)
final_results = {f"{initial_epsilon=}, {min_epsilon=}, {decay_rate=}": {"episode_returns": episode_returns}}
plot_episode_returns(final_results, title="Final Episode Return Convergence")
print_policies_and_q_values(Q, label="Final Policy")