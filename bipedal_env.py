import gymnasium as gym


env= gym.make("BipedalWalker-v3", render_mode="human")
observation, info = env.reset(seed=42)
for _ in range(1000):
    action = env.action_space.sample()  # your agent here (this takes random actions)
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset(seed=42)
env.close()