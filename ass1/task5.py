import numpy as np
import math
from tqdm import tqdm
import pandas as pd

# Define state space $\mathcal{S}$.
states = np.array([
    i for i in range(51)
])

# Define action space, $\mathcal{A}$.
actions = [0,1] 
actions_dict = {
    0: "do nothing",
    1: "do maintenance"
}

# Create state-action cost matrix $\mathcal{C}^{(a)}_{(T,d)}$.
C = np.array([[0, 1]] * 50 + [[np.inf, 5]])

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

pi_zero_infl = 1/2
lambda_zero_infl = 4

def transition_prob0(d1,d2):
    if d2<d1 or d1==50:
        return 0

    if d1==d2:
        return get_zero_prob(pi_zero_infl, lambda_zero_infl)

    elif 0<=d1<15:
        if d2 < 15:
            return get_y_prob(pi_zero_infl, lambda_zero_infl, d2 - d1)
        elif 15 <=d2<30:
            return (2/3) * get_y_prob(pi_zero_infl, lambda_zero_infl, d2 - d1)
        elif 30 <=d2<50:
            return (1/3) * get_y_prob(pi_zero_infl, lambda_zero_infl, d2 - d1)
        elif d2 == 50: # d2 == FAIL
            return (1/3) * sum(
                get_geq_prob(pi_zero_infl, lambda_zero_infl, xi - d1)
                for xi in [15,30,50])
        else:
            print(f'ERROR: {d1} -> {d2}')

    elif 15<=d1<30:
        if d2<30:
            return get_y_prob(pi_zero_infl, lambda_zero_infl, d2 - d1)
        elif 30<=d2<50:
            return (1/2) * get_y_prob(pi_zero_infl, lambda_zero_infl, d2 - d1)
        elif d2==50: # d2 == FAIL
            return (1/2) * sum(
                get_geq_prob(pi_zero_infl, lambda_zero_infl, xi - d1)
                for xi in [30,50])
        else:
            print(f'ERROR: {d1} -> {d2}')

    elif 30<=d1<50:
        if d2<50:
            return get_y_prob(pi_zero_infl, lambda_zero_infl, d2 - d1)
        elif d2==50: # d2 == FAIL
            return get_geq_prob(pi_zero_infl, lambda_zero_infl, 50 - d1)
        else:
            print(f'ERROR: {d1} -> {d2}')
    else:
        print(f'ERROR: {d1} -> {d2}')

P_0 = np.array([
    [transition_prob0(d1,d2) for d2 in states]
    for d1 in states
], dtype=object)

# check if all rows sum to 1 (ignoring floating-point precision erros)
for i,row in enumerate(P_0):
    assert round(sum(row),5) == 1 or round(sum(row),8) == 0, (i, sum(row))


# Get transition probability matrix under action 1; $\mathcal{P}^1$.
P_1 = np.array([
    [1 if d2==0 else 0 for d2 in states]
    for d1 in states
], dtype=object)

for row in P_1:
    assert sum(row)==1, sum(row)


# Create transition prob dictionary
P_dict = {0:P_0, 1:P_1}


# initialize and run value iteration
gamma = 0.9
e = pow(10,-8)
max_iter = pow(10,4)
V = np.zeros(len(states))
pi = np.full(len(states),None)

for _ in tqdm(range(max_iter)):
    V_new = np.copy(V)
    max_diff = 0
    
    for s in states:
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