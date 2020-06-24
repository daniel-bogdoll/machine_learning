#https://towardsdatascience.com/reinforcement-learning-demystified-solving-mdps-with-dynamic-programming-b52c8093c919

import numpy as np
import gym.spaces
from gridworld import GridworldEnv

env = GridworldEnv()

def policy_improvement(env, policy_eval_fn=policy_eval, discount_factor=1.0):
    """
    Policy Improvement Algorithm. Iteratively evaluates and improves a policy
    until an optimal policy is found.
    
    Args:
        env: The OpenAI envrionment.
        policy_eval_fn: Policy Evaluation function that takes 3 arguments:
            policy, env, discount_factor.
        discount_factor: gamma discount factor.
        
    Returns:
        A tuple (policy, V). 
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.
       
    """
    def one_step_lookahead(s, value_fn):

        actions = np.zeros(env.nA)

        for a in range(env.nA):

            [(prob, next_state, reward, done)] = env.P[s][a]
            actions[a] = prob * (reward + discount_factor * value_fn[next_state])
            
        return actions

    # Start with a random policy
    policy = np.ones([env.nS, env.nA]) / env.nA
    actions_values = np.zeros(env.nA)

    while True:

        #evaluate the current policy
        value_fn = policy_eval_fn(policy, env)
       
        policy_stable = True

        #loop over state space
        for s in range(env.nS):


            #perform one step lookahead
            actions_values = one_step_lookahead(s, value_fn)
            
        	#maximize over possible actions 
            best_action = np.argmax(actions_values)

            #best action on current policy
            chosen_action = np.argmax(policy[s])

    		#if Bellman optimality equation not satisifed
            if(best_action != chosen_action):
                policy_stable = False

            #the new policy after acting greedily w.r.t value function
            policy[s] = np.eye(env.nA)[best_action]

        #if Bellman optimality eqn is satisfied
        if(policy_stable):
            return policy, value_fn

    
    

policy, v = policy_improvement(env)
print("Policy Probability Distribution:")
print(policy)
print("")

print("Reshaped Grid Policy (0=up, 1=right, 2=down, 3=left):")
print(np.reshape(np.argmax(policy, axis=1), env.shape))
print("")

print("Value Function:")
print(v)
print("")

print("Reshaped Grid Value Function:")
print(v.reshape(env.shape))
print("")