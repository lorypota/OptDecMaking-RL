import numpy as np
import math
import scipy.stats as stats
from tqdm import tqdm 

# Utility
np.random.seed(0)

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

def print_Q_values(model, states, h):
    state_to_index = {state: idx for idx, state in enumerate(states)}
    
    def Q_value(state, a, h_vec):
        cost, transitions = model[state][a]
        if len(transitions) == 0:
            return cost
        q = cost
        for next_state, prob in transitions:
            q += prob * h_vec[state_to_index[next_state]]
        return q


    print(f"{'State':<20}{'Q(0)':<15}{'Q(1)':<15}")
    
    for state in states:
        q0 = Q_value(state, 0, h) if 0 in model[state] else math.inf
        q1 = Q_value(state, 1, h) if 1 in model[state] else math.inf
        print(f"{str(state):<20}{q0:<15.4f}{q1:<15.4f}")

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

def build_model():
    model = {}  # model[(comp_type, s)] will be a dict: {a: (cost, transitions)}
    states = []
    for comp_type in range(len(xi)):
        for s in range(xi[comp_type] + 1):
            state = (comp_type, s)
            states.append(state)
            model[state] = {}
            
            for a in [0, 1]:
                cost = C[comp_type][s][a]
                
                # Action 0: if state s is below threshold, use the zero-inflated Poisson for increments.
                if a == 0:
                    if s < xi[comp_type]:
                        delta_max = xi[comp_type] - s  # maximum increment before hitting threshold
                        prob_vector = zero_inflated_prob_vector(p_zero, dist_name, (lambda_poisson,), delta_max)
                        transitions = []
                        # For each possible increment (0 to len(prob_vector)-1)
                        for inc, p in enumerate(prob_vector):
                            next_s = s + inc
                            if next_s > xi[comp_type]:
                                next_s = xi[comp_type]
                            next_state = (comp_type, next_s)
                            transitions.append((next_state, p))
                    else:
                        # At the threshold, we force maintenance;
                        transitions = []
                else:
                    # Action 1: state resets to 0 and component type is randomly chosen uniformly among all types.
                    transitions = []
                    n_types = len(xi)
                    prob = 1.0 / n_types
                    for new_comp in range(n_types):
                        next_state = (new_comp, 0)
                        transitions.append((next_state, prob))
                
                model[state][a] = (cost, transitions)
    return model, states

def relative_value_iteration(model, states, max_iterations, tol):
    """
    Relative value iteration for finding the optimal policy that minimizes
    the long-run average cost g and relative value function v:
    - g: long-run average cost
    - v(x): relative value function
    """
    state_to_index = {state: idx for idx, state in enumerate(states)}

    v = np.zeros(len(states)) # relative value function
    
    # Reference state is first state (explicitly set v(N) = 0)
    ref_state = states[0]
    ref_idx = state_to_index[ref_state]
    
    def Q_value(state, a, v_vec):
        """
        Computes the Q(x,a) where in our case R is a cost (negative reward)
        """
        cost, transitions = model[state][a]
        
        if len(transitions) == 0:
            return cost
        
        q = cost
        for next_state, prob in transitions:
            q += prob * v_vec[state_to_index[next_state]]
        
        return q

    for it in tqdm(range(max_iterations), desc="Relative Value Iteration"):
        new_v = np.zeros(len(states))
        
        # For each state, compute the minimal Q over actions
        for idx, state in enumerate(states):
            q0 = Q_value(state, 0, v) if 0 in model[state] else math.inf
            q1 = Q_value(state, 1, v) if 1 in model[state] else math.inf
            new_v[idx] = min(q0, q1) # min since we're minimizing costs

        # Calculate Mi and mi
        diff_v = new_v - v
        mi = np.min(diff_v)
        Mi = np.max(diff_v)
        
        # Compute g estimate using (Mi + mi)/2
        g_estimate = (Mi + mi) / 2
        
        # Subtract baseline
        new_v -= new_v[ref_idx]
        
        if Mi - mi < tol * abs(mi):
            print(f"Converged after {it+1} iterations")
            print(f"Final bounds: mi={mi:.6f}, Mi={Mi:.6f}")
            print(f"Estimated average cost g = {g_estimate:.4f}")
            break
            
        v = new_v
    else:
        print("Warning: Relative value iteration did not converge within the maximum iterations.")
    
    # Derive the optimal policy: for each state, choose the action that minimizes the Q-value.
    policy = {}
    for state in states:
        q0 = Q_value(state, 0, v) if 0 in model[state] else math.inf
        q1 = Q_value(state, 1, v) if 1 in model[state] else math.inf

        comp_type, s = state
        if s == xi[comp_type]:
            best_action = 1
        else:
            best_action = 0 if q0 <= q1 else 1
        policy[state] = best_action

    return g_estimate, v, policy

# Final run
model, states = build_model()
g, v, policy = relative_value_iteration(model, states, max_iterations=100000,  tol=1e-8)

print()
print("Optimal Policy (0 = Do nothing, 1 = Maintenance):")
print_optimal_policy(policy, xi)
print_Q_values(model, states, v)