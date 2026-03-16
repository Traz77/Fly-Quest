from environment import Environment, HUMAN_REWARD
from agent import Agent
import time 
import torch

MAX_EVAL_STEPS = 200

if __name__ == "__main__":
    env = Environment(10, 10)
    agent = Agent(["w", "a", "s", "d"])

    agent.brain.load_state_dict(torch.load("fly_brain.pth", weights_only=True))
    
    agent.epsilon = 0.0

    state = env.get_state()
    done = False 

    print("Game Starting") 
    env.render()
    time.sleep(1)

    steps = 0 
    while not done and steps < MAX_EVAL_STEPS: 
        action = agent.get_action(state)

        next_state, reward, done = env.step(action)
        state = next_state

        print(f"Ation taken: {action}")
        env.render()
        time.sleep(0.5)

        steps += 1

    if not done: 
        print("Times out!")

    if reward == HUMAN_REWARD:
        print("Human was bitten!")
    else:
        print("Fly was eaten!")