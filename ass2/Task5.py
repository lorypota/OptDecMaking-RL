import numpy as np
from tqdm import tqdm 
import matplotlib.pyplot as plt
import scipy.stats as stats
import math

####################################
##  Task 5 - REINFORCE Algorithm  ##
####################################

# Define MDP parameters
actions = (0, 1)  # (0 = do nothing, 1 = maintenance)
xi = (15, 30, 50)  # failure threshold per type of component
S = [list(range(x + 1)) for x in xi]
C = tuple([[0, 1]] * x + [[math.inf, 5]] for x in xi)
gamma = 0.9  # Discount factor

# Utility
np.random.seed(0)

def plot_returns(episode_returns, title="Episode Return Convergence"):
    """
    Utility function to plot episode returns.
    """
    plt.figure(figsize=(12, 7))
    plt.plot(episode_returns)
    plt.xlabel("Episode")
    plt.ylabel("Return (G₀)")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def print_policy_from_theta(theta):
    """
    Prints the greedy policy derived from the learned policy parameters.
    """
    print("Learned Greedy Policy (0 = Do nothing, 1 = Maintenance):")
    for comp_type in range(len(theta)):
        policy = np.argmax(theta[comp_type], axis=0)
        policy[xi[comp_type]] = 1  # force maintenance at threshold
        print(f"Component Type {comp_type + 1} (Threshold = {xi[comp_type]}):")
        print(policy)
    print("\n")


# Transition probabilities
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

def softmax(logits):
    """Compute softmax values for each set of scores in logits."""
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / np.sum(exp_logits)

def run_REINFORCE(nEpisodes, lengthEpisode, initial_lr, decay_rate=5000, patience=None):
    """
    REINFORCE algorithm with baseline variance reduction.
    
    Parameters:
      nEpisodes    : Total number of episodes.
      lengthEpisode: Number of steps per episode.
      initial_lr   : Initial learning rate.
      decay_rate   : Decay rate for the learning rate.
      patience     : (Optional) number of episodes with unchanged policy before early stopping.
    
    Returns:
      theta         : Learned policy parameters.
      episode_returns: List of returns from each episode.
      policy_history: History of greedy policies (for monitoring).
    """
    
    # Initialize policy parameters (theta) for each component type.
    # Each theta[comp_type] is a (2, xi+1) array: one column per state, one row per action.
    theta = tuple(np.zeros((2, x + 1)) for x in xi)
    
    episode_returns = []
    policy_history = []
    stable_count = 0
    prev_policy = None

    for i in tqdm(range(nEpisodes), desc="Episodes"):
        # Decay learning rate over episodes
        lr = initial_lr / (1 + i / decay_rate)
        
        # Generate an episode trajectory: list of (comp_type, s, a, r)
        trajectory = []
        comp_type = np.random.randint(0, len(xi))
        s = 0  # starting state for the chosen component type
        
        for _ in range(lengthEpisode):
            logits = theta[comp_type][:, s]
            probs = softmax(logits)
            
            # At failure threshold, force maintenance but still use softmax for gradient update.
            if s == xi[comp_type]:
                a = 1
            else:
                a = np.random.choice([0, 1], p=probs)
            
            # Simulate the environment transition.
            if a == 0:
                prob_vector = zero_inflated_prob_vector(p_zero, dist_name, (lambda_poisson,), xi[comp_type] - s)
                increments = np.arange(len(prob_vector))
                increment = np.random.choice(increments, p=prob_vector)
                s_prime = s + increment
                s_prime = min(s_prime, xi[comp_type])
                comp_type_prime = comp_type
            else:
                s_prime = 0
                comp_type_prime = np.random.randint(0, len(xi))
            
            r = -C[comp_type][s][a]
            trajectory.append((comp_type, s, a, r))
            
            # Transition to next state.
            s = s_prime
            comp_type = comp_type_prime
        
        # Compute discounted returns G_t (backward pass).
        returns = np.zeros(len(trajectory))
        G = 0
        for t in reversed(range(len(trajectory))):
            _, _, _, r = trajectory[t]
            G = r + gamma * G
            returns[t] = G
        episode_returns.append(returns[0])
        
        # Compute a baseline as the mean return for this episode.
        baseline = np.mean(returns)
        
        # Update policy parameters using the REINFORCE update with baseline.
        for (comp_type, s, a, _), G in zip(trajectory, returns):
            logits = theta[comp_type][:, s]
            probs = softmax(logits)
            # Compute gradient: ∇θ log π(a|s) = one_hot(a) - π(·|s)
            grad_log = np.zeros_like(probs)
            grad_log[a] = 1
            grad_log -= probs
            # Advantage: (G - baseline)
            advantage = G - baseline
            theta[comp_type][:, s] += lr * advantage * grad_log

        # For monitoring: extract the greedy policy (argmax of logits).
        current_policy = []
        for comp in range(len(theta)):
            greedy_policy = np.argmax(theta[comp], axis=0)
            greedy_policy[xi[comp]] = 1  # force maintenance at threshold
            current_policy.append(greedy_policy)
        policy_history.append(tuple(current_policy))
        
        # Optional early stopping if the greedy policy remains unchanged.
        if patience is not None:
            if prev_policy is not None and all(np.array_equal(cp, pp) for cp, pp in zip(current_policy, prev_policy)):
                stable_count += 1
                if stable_count >= patience:
                    print(f"Stopped early at episode {i+1} — policy stable for {patience} episodes.")
                    break
            else:
                stable_count = 0
            prev_policy = current_policy

    return theta, episode_returns, policy_history


# parameters
nEpisodes = 10**5
lengthEpisode = 10**3
initial_lr = 0.05
decay_rate = 10_000

theta, episode_returns, policy_history = run_REINFORCE(nEpisodes, lengthEpisode, initial_lr=initial_lr, decay_rate=decay_rate, patience=200)
plot_returns(episode_returns, title="Final Episode Return Convergence")
print_policy_from_theta(theta)
