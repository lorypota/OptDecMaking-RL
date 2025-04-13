import numpy as np
import math
import scipy.stats as stats
from tqdm import tqdm 

#########################################
##  Task 4 - Relative Value Iteration  ##
#########################################

# Utility
np.random.seed(0)

def print_optimal_policies(policies, xi):
    """Utility function to print policies for each component type"""
    for comp_type, policy in enumerate(policies):
        decisions = [policy.get((comp_type, s), None) for s in range(xi[comp_type]+1)]
        
        print(f"Component Type {comp_type+1}:")
        print("Action:", decisions)
        print()

def print_Q_values(model, states, v_values, xi):
    """Utility function to print Q-values organized by component type"""
    num_comp_types = len(xi)
    
    # Helper function to calculate Q-value
    def calculate_q_value(state, action, v_vec, state_to_index, comp_type):
        cost, transitions = model[state][action]
        if len(transitions) == 0:
            return cost
            
        q = cost
        for next_state, prob in transitions:
            if next_state[0] == comp_type and next_state in state_to_index:
                q += prob * v_vec[state_to_index[next_state]]
        return q
    
    for comp_type in range(num_comp_types):
        # Get states and create index mapping for this component type
        comp_states = [(t, s) for t, s in states if t == comp_type]
        state_to_index = {state: idx for idx, state in enumerate(comp_states)}
        v = v_values[comp_type]
        
        print(f"\nComponent Type {comp_type+1}:")
        print(f"{'State':<10}{'Q(0)':<15}{'Q(1)':<15}")
        
        for s in range(xi[comp_type] + 1):
            state = (comp_type, s)
            if state in state_to_index:
                q0 = calculate_q_value(state, 0, v, state_to_index, comp_type) if 0 in model[state] else math.inf
                q1 = calculate_q_value(state, 1, v, state_to_index, comp_type)
                print(f"({comp_type}, {s})".ljust(10), f"{q0:<15.4f}{q1:<15.4f}")

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
    model = {}  # model[(comp_type, s)]: {a: (cost, transitions)}
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
                        # At the threshold, force maintenance;
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
    results = []
    
    for comp_type in range(3):
        comp_states = [(t, s) for t, s in states if t == comp_type]
        state_to_index = {state: idx for idx, state in enumerate(comp_states)}
        
        v = np.zeros(len(comp_states))  # value function for this component type
        
        def Q_value(state, a, v_vec):
            """Computes the Q(x,a) for this component type"""
            cost, transitions = model[state][a]
            
            if len(transitions) == 0:
                return cost
            
            q = cost
            for next_state, prob in transitions:
                # If next_state is from a different component type, we need special handling
                if next_state[0] != comp_type:
                    # For transitions to other component types, we use 0 as the value
                    # This approximates the interconnected nature of the model
                    continue
                    
                q += prob * v_vec[state_to_index[next_state]]
            
            return q
        
        for it in tqdm(range(max_iterations), desc=f"RVI Component {comp_type+1}"):
            new_v = np.zeros(len(comp_states))
            
            # Compute the minimal Q over actions for each state
            for idx, state in enumerate(comp_states):
                q0 = Q_value(state, 0, v) if 0 in model[state] else math.inf
                q1 = Q_value(state, 1, v)
                new_v[idx] = min(q0, q1)
            
            # Calculate Mi and mi
            diff_v = new_v - v
            mi = np.min(diff_v)
            Mi = np.max(diff_v)
            
            # Compute g estimate
            g_estimate = (Mi + mi) / 2
            
            # Normalize on (comp_type, 0)
            ref_state = (comp_type, 0)
            ref_idx = state_to_index[ref_state]
            ref_val = new_v[ref_idx]
            new_v = new_v - ref_val
            
            if Mi - mi < tol * abs(mi):
                print(f"Component {comp_type+1} converged after {it+1} iterations")
                print(f"Final bounds: mi={mi:.6f}, Mi={Mi:.6f}")
                print(f"Estimated average cost g_{comp_type+1} = {g_estimate:.4f}")
                break
                
            v = new_v
        
        # Derive optimal policy
        policy = {}
        for state in comp_states:
            q0 = Q_value(state, 0, v) if 0 in model[state] else math.inf
            q1 = Q_value(state, 1, v)
            
            comp_type, s = state
            if s == xi[comp_type]:
                best_action = 1
            else:
                best_action = 0 if q0 <= q1 else 1
            policy[state] = best_action
        
        # Store results
        results.append((g_estimate, v, policy))
    
    g_values = [r[0] for r in results]
    v_values = [r[1] for r in results]
    policies = [r[2] for r in results]
    
    return g_values, v_values, policies

# Final run
model, states = build_model()
g_values, v_values, policies = relative_value_iteration(model, states, max_iterations=100000, tol=1e-8)

print("\nLong-run average costs:")
for comp_type in range(3):
    print(f"* g for component type {comp_type+1} is: {g_values[comp_type]:.4f}")
print()

print_optimal_policies(policies, xi)

print_Q_values(model, states, v_values, xi)