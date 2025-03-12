import numpy as np
from tqdm import tqdm
import math

### States, action space and cost (reward) ###

# degradation thresholds
xi = {1: 15, 2: 30, 3: 50}

# State space
S = []
for T in [1, 2, 3]:
    for d in range(xi[T]+1):
        S.append((T, d))

# Get total number of states
num_states = len(S)
print(f"There are {len(S)} states in our MDP.")

# Action space
A = [0, 1]  # 0: "Do nothing", 1: "Do maintenance"

# Cost function C[s][a] where s is current state, a is action
C = np.zeros((num_states, 2))


for idx_s, s in enumerate(S):
    T, d = s

    if d == xi[T]:
        C[idx_s][0] = float("inf")
        C[idx_s][1] = 5
    else:
        C[idx_s][0] = 0
        C[idx_s][1] = 1



## Transition probabilities
lambda_param = 4
pi = 0.5

def zero_inflated_poisson_pmf(k, lambda_param=lambda_param, pi=pi):
    if k == 0:
        return pi + (1 - pi) * np.exp(-lambda_param)
    else:
        return (1 - pi) * (np.power(lambda_param, k) * np.exp(-lambda_param)) / math.factorial(k)
    
# Calculate the maximum k needed for zero-inflated Poisson
def find_max_k(lambda_param=lambda_param, pi=pi, epsilon=np.finfo(float).eps):
    k = 0
    while True:
        # Calculate probability for this k
        prob = zero_inflated_poisson_pmf(k, lambda_param, pi)
        
        # If probability is below floating-point precision, we've found our cutoff
        if prob < epsilon:
            return k
        k += 1

# Calculate the maximum k needed once
max_k = find_max_k() # = 30
print(f"Maximum k needed for sufficient precision: {max_k}")


# P[a][s][s'] where a is action, s is current state, s' is next state
P = np.zeros((2, num_states, num_states))

# Calculate P0 (Do nothing)
for idx_s, s in enumerate(S):
    T, d = s
    
    # Cannot do nothing in failed state
    if d == xi[T]:
        continue
        
    for idx_s_prime, s_prime in enumerate(S):
        T_prime, d_prime = s_prime
        
        # Type cannot change under "do nothing"
        if T != T_prime:
            continue
            
        # Calculate transition probability based on degradation increase
        if d <= d_prime < xi[T]:
            P[0][idx_s][idx_s_prime] = zero_inflated_poisson_pmf(d_prime - d)
        elif d < xi[T] and d_prime == xi[T]:
            # Transition to failed state (cumulative probability of large increases)
            cumulative_prob = 0
            for k in range(xi[T] - d, max_k):
                cumulative_prob += zero_inflated_poisson_pmf(k)
            P[0][idx_s][idx_s_prime] = cumulative_prob

# Calculate P1 (Do maintenance)
for idx_s, s in enumerate(S):
    for idx_s_prime, s_prime in enumerate(S):
        (T_prime, d_prime) = s_prime
        if d_prime == 0:
            P[1][idx_s][idx_s_prime] = 1/3

print(f"Transition probability matrices created with shape: {P.shape}")


# Policy iteration
gamma = 0.9

# Initialize policy randomly
policy = np.zeros(num_states, dtype=int)
for s in range(num_states):
    if C[s][0] == float("inf"):
        policy[s] = 1  # Use action 1 if action 0 is unavailable

old_policy = np.ones_like(policy)
iterations = 0

while iterations == 0 or np.any(policy != old_policy):
    iterations += 1
    old_policy = policy.copy()

    # Step 1
    P_pi = np.zeros((num_states, num_states))
    C_pi = np.zeros(num_states)
    for s in range(num_states):
        a = policy[s]
        P_pi[s, :] = P[a][s]
        C_pi[s] = C[s][a]
    
    I = np.identity(num_states)
    V = np.linalg.inv(I - gamma * P_pi) @ C_pi

    # Step 2
    for s in range(num_states):
        min_cost = float("inf")
        best_action = policy[s]
        
        for a in range(2):
            if C[s][a] == float("inf"):
                continue  # Skip unavailable actions

            expected_cost = C[s][a] + gamma * np.sum(P[a][s] * V)
            
            if expected_cost < min_cost:
                min_cost = expected_cost
                best_action = a
        
        policy[s] = best_action
    
print(f"Converged at iteration {iterations}")
print(f"Best policy {policy}")


print("\n=== Expected Discounted Costs for All States ===")

# Type 1 states
print("\nType 1 Component:")
for d in range(16):  # Type 1 has states 0-15
    state_idx = S.index((1, d))
    print(f"State (1,{d}): {V[state_idx]:.4f}")

# Type 2 states
print("\nType 2 Component:")
for d in range(31):  # Type 2 has states 0-30
    state_idx = S.index((2, d))
    print(f"State (2,{d}): {V[state_idx]:.4f}")

# Type 3 states
print("\nType 3 Component:")
for d in range(51):  # Type 3 has states 0-50
    state_idx = S.index((3, d))
    print(f"State (3,{d}): {V[state_idx]:.4f}")