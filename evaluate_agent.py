from environment import Environment, HUMAN_REWARD, SPIDER_PENALTY
from agent import Agent
import torch

# Evaluation constants
EVAL_EPISODES = 10000
GRID_SIZE = 10
MAX_STEPS = 200

if __name__ == "__main__":
    env = Environment(GRID_SIZE, GRID_SIZE)
    agent = Agent(["w", "a", "s", "d"])

    # Load the trained brain
    try:
        agent.brain.load_state_dict(torch.load("fly_brain.pth", weights_only=True))
        print("Trained brain loaded successfully!")
    except FileNotFoundError:
        print("Error: fly_brain.pth not found. Please train the agent first.")
        exit()

    # Set epsilon to 0.0 for evaluation (greedy policy)
    agent.epsilon = 0.0

    print(f"Starting evaluation over {EVAL_EPISODES} episodes...")
    
    wins = 0
    deaths = 0
    timeouts = 0

    for episode in range(EVAL_EPISODES):
        env = Environment(GRID_SIZE, GRID_SIZE)
        state = env.get_state()
        done = False
        steps = 0

        while not done and steps < MAX_STEPS:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            state = next_state
            steps += 1

        if done and reward == HUMAN_REWARD:
            wins += 1
        elif done and reward == SPIDER_PENALTY:
            deaths += 1
        else:
            timeouts += 1

        if (episode + 1) % 1000 == 0:
            print(f"Evaluated {episode + 1}/{EVAL_EPISODES} episodes...")

    win_rate = (wins / EVAL_EPISODES) * 100
    
    print("\n" + "="*30)
    print("      EVALUATION RESULTS")
    print("="*30)
    print(f"Total Episodes: {EVAL_EPISODES}")
    print(f"Wins:           {wins}")
    print(f"Spider Deaths:  {deaths}")
    print(f"Timeouts:       {timeouts}")
    print(f"Win Rate:       {win_rate:.2f}%")
    print("="*30)
