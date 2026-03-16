from metrics import (start_metrics_server, episodes_total, wins_total,
                     spider_deaths_total, timeouts_total, episode_reward, 
                     episode_length)
from environment import Environment, HUMAN_REWARD, SPIDER_PENALTY
from agent import Agent
import torch

# Training constants
EPISODES = 30000
GRID_SIZE = 10
MAX_STEPS = 200
TIMEOUT_PENALTY = -10
TRAIN_EVERY = 4
EPSILON_FLOOR = 0.05
PRINT_EVERY = 100

if __name__ == "__main__":
    env = Environment(GRID_SIZE, GRID_SIZE)
    agent = Agent(["w", "a", "s", "d"])

    episodes = EPISODES
    start_metrics_server()
    
    for episode in range(episodes):
        env = Environment(GRID_SIZE, GRID_SIZE)
        state = env.get_state()
        done = False
        total_reward = 0

        steps = 0
        max_steps = MAX_STEPS
        while not done and steps < max_steps:
            action = agent.get_action(state)
            
            # Take the step in the game 
            next_state, reward, done = env.step(action)
            total_reward += reward

            agent.remember(state, action, reward, next_state, done)
            
            # Learn from the new data of the new state 
            if steps % TRAIN_EVERY == 0:
                agent.train(state, action, reward, next_state, done)

            state = next_state

            steps += 1
        
        if not done: 
            agent.train(state, action, TIMEOUT_PENALTY, next_state, True)

        episodes_total.inc()          
        episode_reward.set(total_reward)  # this episode's total reward
        episode_length.set(steps)     # how many steps this episode took
        # Classify the outcome
        if done and reward == HUMAN_REWARD:
            wins_total.inc()
        elif done and reward == SPIDER_PENALTY:
            spider_deaths_total.inc()
        else:
            timeouts_total.inc()

        if agent.epsilon > EPSILON_FLOOR:
            agent.epsilon *= agent.epsilon_decay
        
        if episode % PRINT_EVERY == 0:
            print(f"Episode {episode}/{episodes} | Epsilon: {agent.epsilon:.3f}")
    
    torch.save(agent.brain.state_dict(), "fly_brain.pth")
    print ("Brain saved successfully!")