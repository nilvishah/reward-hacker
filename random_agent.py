import gymnasium as gym

env = gym.make("LunarLander-v3", render_mode="human")
obs, info = env.reset(seed=0)

for step in range(1000):
    action = env.action_space.sample()  # random action
    obs, reward, terminated, truncated, info = env.step(action)
    if step % 50 == 0:
        print(step, [round(x, 2) for x in obs], round(reward, 2))
    if terminated or truncated:
        obs, info = env.reset()

env.close()