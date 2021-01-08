import gym
import numpy as np

env = gym.make("MountainCar-v0")

LEARNING_RATE = 0.1

DISCOUNT = 0.95
EPISODES = 25000
SHOW_EVERY = 500
solved_episodes = 0

#Actual states are continuous, but this would result in an infinite amount of states where no q-table would be possible, thus the state space is being discretized
DISCRETE_OS_SIZE = [20, 20]
discrete_os_win_size = (env.observation_space.high - env.observation_space.low)/DISCRETE_OS_SIZE

# Exploration settings
epsilon = 1  # between 0 and 1, 1 always exploring
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2 #division to integer
epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)

#Q-Table is a lookup table for q-values at each possible discretized state (position & velocity) in the environment. 
q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n])) #gym env sets reward to -1 as default and to 0 for reaching the flag

def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(np.int))  # 3 Q values for the available actions (right, left, nothing?) in the q-table

for episode in range(EPISODES):
    discrete_state = get_discrete_state(env.reset())
    done = False

    if episode % SHOW_EVERY == 0:
        render = True
        print(episode, epsilon, solved_episodes)
        solved_episodes = 0
    else:
        render = False

    while not done:
        if np.random.random() > epsilon:    #q-table action
            action = np.argmax(q_table[discrete_state])
        else:   #random action
            action = np.random.randint(0, env.action_space.n)

        new_state, reward, done, _ = env.step(action)   #Each env has it's own termination rules, here only 200 steps are allowed: https://github.com/openai/gym/blob/master/gym/envs/classic_control/mountain_car.py
        new_discrete_state = get_discrete_state(new_state)

        if render == True:
            env.render()

        # If simulation did not end yet after last step - update Q table
        if not done:

            # Maximum possible Q value in next step (for new state)
            max_future_q = np.max(q_table[new_discrete_state])

            # Current Q value (for current state and performed action)
            current_q = q_table[discrete_state + (action,)]

            # And here's our equation for a new Q value for current state and action
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

            # Update Q table with new Q value
            q_table[discrete_state + (action,)] = new_q


        # Simulation ended (for any reson) - if goal position is achived - update Q value with reward directly
        elif new_state[0] >= env.goal_position: #Couldn't we just check for reward == 0?
            q_table[discrete_state + (action,)] = 0
            solved_episodes += 1
            #print(f"Goal reached in {episode}")

        discrete_state = new_discrete_state

    # Decaying is being done every episode if episode number is within decaying range
    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value

env.close()