from entity import Spider, Fly, Human
import random

# Reward constants
SPIDER_PENALTY = -10
HUMAN_REWARD = 20
WALL_PENALTY = -2
CLOSER_REWARD = -0.5
FARTHER_PENALTY = -1.5
NUM_SPIDERS = 3

class Environment: 
    def __init__(self, x_length, y_length):
        self.x_length = x_length
        self.y_length = y_length
        
        # 1. Spawn Fly
        self.fly = Fly(random.randint(0, x_length - 1), random.randint(0, y_length - 1))
        
        # 2. Spawn Human away from Fly
        while True:
            self.human = Human(random.randint(0, x_length - 1), random.randint(0, y_length - 1))
            if (self.human.x, self.human.y) != (self.fly.x, self.fly.y):
                break
                
        # 3. Spawn 3 Spiders away from both
        self.spiders = []
        for _ in range(NUM_SPIDERS):
            while True:
                sx = random.randint(0, x_length - 1)
                sy = random.randint(0, y_length - 1)
                # Check it doesn't overlap with Fly, Human, or other Spiders
                if (sx, sy) != (self.fly.x, self.fly.y) and \
                   (sx, sy) != (self.human.x, self.human.y) and \
                   not any(s.x == sx and s.y == sy for s in self.spiders):
                    self.spiders.append(Spider(sx, sy))
                    break

    def step(self, action): 
        self.human.move_randomly(self.x_length, self.y_length, self.get_occupied_positions(exclude=self.human))
        
        # Save old distance for reward shaping
        old_distance = abs(self.fly.x - self.human.x) + abs(self.fly.y - self.human.y)
        
        dx, dy = 0, 0
        if action == "w":   dy = -1
        elif action == "s": dy = 1
        elif action == "a": dx = -1
        elif action == "d": dx = 1

        new_x = self.fly.x + dx
        new_y = self.fly.y + dy

        if 0 <= new_x <= self.x_length - 1 and 0 <= new_y <= self.y_length - 1:
            self.fly.move(dx,dy)
        else:
            return self.get_state(), WALL_PENALTY, False
            
        for spider in self.spiders:
            if self.fly.x == spider.x and self.fly.y == spider.y:
                return self.get_state(), SPIDER_PENALTY, True

        if self.fly.x == self.human.x and self.fly.y == self.human.y:
            return self.get_state(), HUMAN_REWARD, True
        
        # Reward shaping: encourage moving closer to Human
        new_distance = abs(self.fly.x - self.human.x) + abs(self.fly.y - self.human.y)
        reward = CLOSER_REWARD if new_distance < old_distance else FARTHER_PENALTY
        
        return self.get_state(), reward, False

    def render(self): 
        for y in range(self.y_length):
            for x in range(self.x_length):
                entity_here = self._entity_at(x, y)
                print(entity_here.symbol if entity_here else ".", end=" ")
            print()

    def get_state(self):
        state =[self.fly.x, self.fly.y, self.human.x, self.human.y]
        for spider in self.spiders:
            state.append(spider.x)
            state.append(spider.y)
        return tuple(state)

    def get_occupied_positions(self, exclude=None):
        all_entities = [self.fly, self.human] + self.spiders
        return [(e.x, e.y) for e in all_entities if e != exclude]

    def _entity_at(self, x, y):
        all_entities = [self.fly, self.human] + self.spiders
        for e in all_entities:
            if e.x == x and e.y == y:
                return e 
        return None 