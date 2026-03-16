from environment import Environment
import csv 
import math
import random

def collect_data(iterations):
    env = Environment(10, 10)

    with open("dataset.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["FlyX", "FlyY", "HumanX", "HumanY", "Distance"])

        for i in range(iterations):
            actions = ["w", "a", "s", "d"]
            action = random.choice(actions)

            state, reward, done = env.step(action)

            fx, fy = env.fly.x, env.fly.y
            hx, hy = env.human.x, env.human.y
            distance = math.sqrt((fx - hx) ** 2 + (fy - hy) ** 2)
            writer.writerow([fx, fy, hx, hy, distance])

            if done:
                env = Environment(10, 10)