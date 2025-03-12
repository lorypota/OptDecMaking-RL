import numpy as np
from tqdm import tqdm
import math

# Define state space $\mathcal{S}$.
states = [
    (i+1,d)
    for i,t in enumerate([15,30,50])
        for d in range(t+1)
]

id_to_state = {i:s for i,s in enumerate(states)}
state_to_id = {s:i for i,s in id_to_state.items()}
print(f"There are {len(states)} states in our MDP.")

# Define action space, $\mathcal{A}$.
actions = [0,1] 
actions_dict = {
    0: "do nothing",
    1: "do maintenance"
}

#Create state-action cost matrix $\mathcal{C}^{(a)}_{(T,d)}$.
thresholds = {1:15,2:30,3:50}

C = np.array([
    [0 if d < thresholds[T] else np.inf, 
     1 if d < thresholds[T] else 5]
    for T, d in states
])

# Get transition probability matrix under action 0; $\mathcal{P}^0$.
def get_zero_prob(pi,lambda_):
    """
    Returns the probability of a zero-inflated Poisson random variable being equal
    to zero: P(P_t+1 = 0). Here pi represents the probability of getting P_t+1 = 0
    deterministically, and (1-pi) represents of drawing from a Poisson distribution.
    """
    return pi + (1-pi)*np.exp(-lambda_)

def get_y_prob(pi,lambda_,y):
    """
    Returns the probability of a zero-inflated Poisson random variable being equal
    to y: P(P_t+1 = y).
    """
    return (1-pi) * ((pow(lambda_,y)*np.exp(-lambda_)) / math.factorial(y))

def get_geq_prob(pi,lambda_,k):
    """
    Returns the probability of a zero-inflated Poisson random variable being
    greater than or equal to k: P(P_t+1 >= k) = 1 - P(P_t+1 < k), where
    k = xi_T - d_t.
    """
    if k == 0:
        return 1.0
    # prob_less_than_k = sum(get_y_prob(pi,lambda_,i) for i in range(0,k))
    prob_less_than_k = get_zero_prob(pi,lambda_) + sum(get_y_prob(pi,lambda_,i) for i in range(1,k))
    return 1 - prob_less_than_k

def transition_prob0(T1, d1, T2, d2):
    """Get transition probability for (T,d) -> (T',d')."""
    th1 = thresholds[T1]
    if d1 < th1:
        if T1!=T2 or d2 < d1:
            return 0
        elif d1 == d2:
            return get_zero_prob(pi_zero_infl, lambda_zero_infl)
        elif d2 < th1:
            # P(P_{t+1} = d2 - d1)
            return get_y_prob(pi_zero_infl, lambda_zero_infl, d2 - d1)
        elif d2 == th1:
            # P(P_{t+1} >= th1 - d1)
            return get_geq_prob(pi_zero_infl, lambda_zero_infl, th1 - d1)
        else:
            print(f'ERROR: {(T1,d1)} -> {(T2,d2)}')
            return None
        
    else:
        return 0


pi_zero_infl = 1/2
lambda_zero_infl = 4

P_0 = np.array([
    [transition_prob0(T1, d1, T2, d2) for (T2, d2) in states]
    for (T1, d1) in states
], dtype=object)

for i,row in enumerate(P_0):
    assert round(sum(row),8) == 1 or round(sum(row),8) == 0, (i, sum(row))


# Get transition probability matrix under action 1; $\mathcal{P}^1$.
def transition_prob1(T1,d1,T2,d2):
    if T1 not in [1,2,3] or T2 not in [1,2,3]:
        print(f'ERROR: {(T1,d1)} -> {(T2,d2)}')
        return None
    if d2 == 0:
        return 1/3
    else:
        return 0

P_1 = np.array([
    [transition_prob1(T1, d1, T2, d2) for (T2, d2) in states]
    for (T1, d1) in states
], dtype=object)

for i,row in enumerate(P_1):
    assert round(sum(row),8) == 1 or round(sum(row),8) == 0, (i, sum(row)) 

# Create transition prob dictionary
P_dict = {0:P_0, 1:P_1}

# Initialize discount factor ($\gamma$), the convergence threshold ($\epsilon$), the maximum number
# of iterations (max\_iter), $\pi_0(s)$, and set $V_0(s)$ to $0$ for all $s\in\mathcal{S}$. Then we perform Value Iteration.
gamma = 0.9
e = pow(10,-8)
max_iter = pow(10,4)
V = np.zeros(len(states))
pi = np.full(len(states),None)

for _ in tqdm(range(max_iter)):
    V_new = np.copy(V)
    max_diff = 0
    
    for (T,d) in states:
        s = state_to_id[(T,d)]
        value_function = []
        for a in actions:
            val = C[s][a] + gamma * np.dot(P_dict[a][s],V)#np.dot(probs[s, :, a], V)
            value_function.append(val)

        V_new[s] = min(value_function)
        pi[s] = np.argmin(value_function)
        max_diff = max(max_diff, abs(V_new[s] - V[s]))
    
    V = V_new
    if max_diff < e:
        break

print("\nOptimal Values:\n", V)
print("\nOptimal Policy:\n", pi)