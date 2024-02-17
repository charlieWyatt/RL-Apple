from stable_baselines3 import PPO
from collect_dots_env import CollectDotsEnv



# Function to run the simulation
def run_simulation(env, model, num_episodes=10, max_steps=1000):
    env.set_max_steps(max_steps)
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        dots_collected = 0
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, done, info = env.step(action)
            env.render()
            if 'dots_collected' in info:
                dots_collected = info['dots_collected']
        print(f"Episode {episode + 1}: Dots collected = {dots_collected}")

if __name__ == "__main__":
    # Load the environment
    env = CollectDotsEnv()

    # Load the pre-trained agent
    model = PPO.load("example_model")
    
    # Run the simulation indefinitely (or for a specific number of episodes)
    run_simulation(env, model, num_episodes=999)

