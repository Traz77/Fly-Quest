import random

class Entity:
    def __init__(self, x, y, symbol):
        self.x = x
        self.y = y
        self.symbol = symbol
    
    def move(self, dx, dy):
        self.x += dx
        self.y += dy

class Fly(Entity):
    def __init__(self, x, y): 
        super().__init__(x, y, "F")

class Human(Entity):
    def __init__(self, x, y): 
        super().__init__(x, y, "H")

    def move_randomly(self, max_x, max_y, blocked_positions=None):
        if blocked_positions is None:
            blocked_positions = []
        dx, dy = random.choice([(0,1), (0,-1), (1,0), (-1,0)])
        new_x = self.x + dx
        new_y = self.y + dy
        if 0 <= new_x < max_x and 0 <= new_y  < max_y \
            and (new_x, new_y) not in blocked_positions:
            self.x = new_x
            self.y = new_y

class Spider(Entity): 
    def __init__(self, x, y):
        super().__init__(x, y, "S")