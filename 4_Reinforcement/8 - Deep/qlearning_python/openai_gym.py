import gym

env = gym.make("MountainCar-v0")
env.reset()

done = False

while not done:
    action = 2  #in MountainCar: go right
    new_state, reward, done, extra_info = env.step(action)
    env.render()

env.close()