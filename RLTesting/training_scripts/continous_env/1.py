import gymnasium as gym

env = gym.make('Pendulum-v1')
env.reset()

action = env.action_space.sample()
result1, result2,result3,result4,result5 = env.step(action)

print(result1, result2, result3, result4, result5)