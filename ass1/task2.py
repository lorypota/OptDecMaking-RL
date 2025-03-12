import numpy as np
from tqdm import tqdm
import random
import math
from numba import njit

# Get all states
states = [
    (i+1,d)
    for i,t in enumerate([15,30,50])
        for d in range(t+1)
]

id_to_state = {i:s for i,s in enumerate(states)}
state_to_id = {s:i for i,s in id_to_state.items()}
print(f"There are {len(states)} states in our MDP.")

# Get cost vector $\mathcal{C}_\pi$, whose $i$'th component is given by $\mathcal{C}_\pi(s_i) = c(s_i,\pi(s_i))$, which represents the immediate cost of taking the policy’s chosen action in state $s_i$.
thresholds = {1:15,2:30,3:50}

C_pi = np.array([
    5 if d == thresholds[T]
      else 0
    for (T, d) in states
], dtype=float)

# Compute transition probability matrix $\mathcal{P}_\pi$.
def get_zero_prob(pi: float, lambda_: float) -> float:
    """
    Returns the probability of a zero-inflated Poisson random variable being equal
    to zero: P(P_t+1 = 0). Here pi represents the probability of getting P_t+1 = 0
    deterministically, and (1-pi) represents of drawing from a Poisson distribution.
    """
    return pi + (1-pi)*np.exp(-lambda_)

def get_y_prob(pi: float,lambda_: float, y: int) -> float:
    """
    Returns the probability of a zero-inflated Poisson random variable being equal
    to y: P(P_t+1 = y).
    """
    return (1-pi) * ((pow(lambda_,y)*np.exp(-lambda_)) / math.factorial(y))

def get_geq_prob(pi: float, lambda_: float, k: int) -> float:
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

P = []
for s1 in states:
    p_row = []
    for s2 in states:
        T1,d1 = s1
        T2,d2 = s2
        
        xiT1 = thresholds[T1]

        # do nothing action
        if d1 < xiT1:
            # impossible cases have probability 0
            if T1!=T2 or d2<d1:
                p = 0

            # probability of staying in the same state
            elif d1 == d2:
                p = get_zero_prob(pi_zero_infl,lambda_zero_infl)

            # probability of going to an increased state
            elif d2 < xiT1: # d1 < d2 < xiT1 (bc we already compared d1 == d2 it is d1<d2)
                p = get_y_prob(pi_zero_infl,lambda_zero_infl,d2-d1)

            # probability of going to the threshold state
            elif d2 == xiT1:
                p = get_geq_prob(pi_zero_infl,lambda_zero_infl,xiT1-d1)
            
            else:
                print(f'ERROR {s1=} -> {s2=}')

        # do maintenance action
        else:
            if d1==xiT1 and d2==0:
                p = 1/3
            else:
                p = 0
        

        p_row.append(p)
    P.append(p_row)
    
P_pi = np.array(P)

for row in P_pi:
    assert round(sum(row),8)==1, sum(row)

# Solve the system using $V_\pi = (I-\gamma \mathcal{P}_\pi)\mathcal{C}_\pi$.
I = np.eye(len(states))
gamma = 0.9

v_pi = np.linalg.solve(I - gamma*P_pi, C_pi)

# print("Computed value function:")
for s,v in zip(states,v_pi):
    print(f"v({tuple(s)}) = {v}")
    break

## ====== TASK 2.1 ======
@njit
def create_cdfs(P_pi):
    """
    Precompute the per-state cumulative distribution function.
    P_pi is assumed to be shape (nr_states, nr_states).
    """
    nr_states = P_pi.shape[0]
    P_cdf = np.empty_like(P_pi)
    for s in range(nr_states):
        # compute cumulative sum for row s
        cdf_row = 0.0
        for sp in range(nr_states):
            cdf_row += P_pi[s, sp]
            P_cdf[s, sp] = cdf_row
    return P_cdf

@njit
def simulate_episode_cdf(s0, P_cdf, C_pi, discounts, n):
    """
    Simulate one episode of length n starting at state s0 using
    the precomputed CDF matrix P_cdf. Then compute the
    discounted return from cost vector C_pi.
    """
    sim = np.empty(n, dtype=np.int64)
    sim[0] = s0
    # generate all random numbers needed at once
    us = np.random.rand(n - 1)
    for t in range(1, n):
        # next state from P_cdf of the current state
        sim[t] = np.searchsorted(P_cdf[sim[t - 1]], us[t - 1])
    return np.sum(discounts * C_pi[sim])

def monte_carlo_mdp_fast(
    P_pi, 
    C_pi, 
    state_to_id, 
    n=100_000, 
    num_episodes=100, 
    gamma=0.9
):
    s0 = state_to_id[(1,0)]  # example start
    nr_states = len(P_pi)

    # Precompute
    P_cdf = create_cdfs(P_pi)  # shape (nr_states, nr_states)
    discounts = gamma ** np.arange(n)

    returns = np.empty(num_episodes)
    for i in tqdm(range(num_episodes)):
        G = simulate_episode_cdf(s0, P_cdf, C_pi, discounts, n)
        returns[i] = G

    return returns.mean(),returns

gamma = 0.9
n = 10_000
num_episodes = 100
v1_0_estimate100,sample100 = monte_carlo_mdp_fast(P_pi, C_pi, state_to_id,n=n,num_episodes=num_episodes)
print(f"Estimated v_pi((1,0)) over {num_episodes} episodes of length {n}  = {v1_0_estimate100:.4f}")

num_episodes = 1_000
v1_0_estimate1k,sample1k = monte_carlo_mdp_fast(P_pi, C_pi, state_to_id,n=n,num_episodes=num_episodes)
print(f"Estimated v_pi((1,0)) over {num_episodes} episodes of length {n}  = {v1_0_estimate1k:.4f}")

num_episodes = 10_000
v1_0_estimate10k,sample10k = monte_carlo_mdp_fast(P_pi, C_pi, state_to_id,n=n,num_episodes=num_episodes)
print(f"Estimated v_pi((1,0)) over {num_episodes} episodes of length {n}  = {v1_0_estimate10k:.4f}")


# ===== CI =====
def get_sample_variance(sample):
    n = len(sample)
    avg = np.mean(sample)
    return (1/n) * sum((sample-avg)**2)

def get_ci(estimate,z,S2,n):
    half_width = z * (S2/n)**0.5
    return (estimate - half_width, p+half_width)

def get_half_width(z,S2,n):
    return z * (S2/n)**0.5

S2_100 = 1.157914873493865 #get_sample_variance(sample100)
S2_1k = 1.0486945991784795 #get_sample_variance(sample1k)
S2_10k = 1.0677367537314746 #get_sample_variance(sample10k)

z = 1.96

print(f"95% CI for 100 episodes: {get_ci(v1_0_estimate100,z,S2_100,100)}\nhalf-width: {get_half_width(z,S2_100,100)}\n")
print(f"95% CI for 1k episodes: {get_ci(v1_0_estimate1k,z,S2_1k,1000)}\nhalf-width: {get_half_width(z,S2_1k,1_000)}\n")
print(f"95% CI for 10k episodes: {get_ci(v1_0_estimate10k,z,S2_10k,10_000)}\nhalf-width: {get_half_width(z,S2_10k,10_000)}")