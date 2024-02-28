import gymnasium as gym
import numpy as np

env = gym.make('Pendulum-v1')


incorrect_count = 0
for _ in range(100):
    state = env.observation_space.sample()

    cos_theta = state[0]
    sin_theta = state[1]

    sum_of_squares = cos_theta**2 + sin_theta**2

    print(f"Sum of squares: {sum_of_squares}")
    if np.isclose(sum_of_squares, 1.0, atol=0.1):
        print("Sample is correct.")
    else:
        print("Sample is incorrect.")
        incorrect_count += 1

print(incorrect_count)