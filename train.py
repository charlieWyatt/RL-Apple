from stable_baselines3 import PPO
from collect_dots_env import CollectDotsEnv
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback

class DotCollectionLogger(BaseCallback):
    def __init__(self, verbose=0):
        super(DotCollectionLogger, self).__init__(verbose)
        self.dots_collected = 0
        self.dots_collected_per_episode = []

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self):
        # This method is called at the end of a rollout, before updating the policy
        # Check if 'dots_collected' is in the info dict and log it
        episode_dots = sum(info.get('dots_collected', 0) for info in self.locals["infos"])
        self.dots_collected_per_episode.append(episode_dots)
        print(f"Dots collected in the last episode: {episode_dots}")

def train():
    # Create the environment
    env = CollectDotsEnv()

    # # Initialize the agent
    model = PPO("MlpPolicy", env, verbose=1)
    model.load("example_model")

    # Initialize the callback
    dot_collection_logger = DotCollectionLogger()

    # Train the agent
    model.learn(total_timesteps=1000000, callback=dot_collection_logger)

    model.save("new_model")

    # Plotting the results
    plt.figure(figsize=(10, 5))
    plt.plot(dot_collection_logger.dots_collected_per_episode)
    plt.xlabel('Episode')
    plt.ylabel('Dots Collected')
    plt.title('Dots Collected Over Episodes')
    plt.savefig('dots_collected_over_episodes.png')  # Save the figure
    plt.show()  # Display the plot

    # Test the trained agent
    obs = env.reset()
    for _ in range(100000):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        env.render()
        if dones:
            episode_dots_collected = info["dots_collected"]  # Get the count from the environment
            print(f"Episode ended. Dots collected: {episode_dots_collected}")
            obs = env.reset()

if __name__ == "__main__":
    train()