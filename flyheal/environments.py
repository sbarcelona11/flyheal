"""Seeded continuous 2D foraging with eight local sensory observations."""
import math
import gymnasium as gym
import numpy as np
from gymnasium import spaces


class ForagingEnv(gym.Env):
    metadata = {'render_modes': []}
    width = height = 10.0
    radius = .15
    food_radius = .3
    speed = .18
    turn = .35
    horizon = 500
    sensor_range = 4.0

    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(-1., 1., shape=(8,), dtype=np.float32)

    def _blocked(self, x, y):
        r = self.radius
        if not (r <= x <= self.width-r and r <= y <= self.height-r):
            return True
        for ox, oy, w, h in self.obstacles:
            # Exact circle/rectangle overlap; movement is shorter than the radius diameter.
            cx, cy = min(max(x, ox), ox+w), min(max(y, oy), oy+h)
            if (x-cx)**2 + (y-cy)**2 <= r*r:
                return True
        return False

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.position = np.array([5., 5.], dtype=float)
        self.heading = float(self.np_random.uniform(-math.pi, math.pi))
        self.obstacles = []
        for _ in range(5):
            for attempt in range(100):
                w, h = self.np_random.uniform(.5, 1.15, 2)
                x, y = self.np_random.uniform(.6, 8.2, 2)
                # Keep the start clear and obstacles separate with a navigable gap.
                if x-1 < 5 < x+w+1 and y-1 < 5 < y+h+1:
                    continue
                if any(x < ox+ow+.6 and x+w+.6 > ox and y < oy+oh+.6 and y+h+.6 > oy
                       for ox, oy, ow, oh in self.obstacles):
                    continue
                self.obstacles.append((float(x), float(y), float(w), float(h)))
                break
        foods = []
        while len(foods) < 3:
            point = self.np_random.uniform(.65, 9.35, 2)
            if self._blocked(*point) or np.linalg.norm(point-self.position) < 1.2:
                continue
            if any(np.linalg.norm(point-p) < 1. for p in foods):
                continue
            # Food must not overlap an obstacle even at its outer edge.
            if any(ox-.5 < point[0] < ox+w+.5 and oy-.5 < point[1] < oy+h+.5
                   for ox, oy, w, h in self.obstacles):
                continue
            foods.append(point)
        self.foods = np.array(foods)
        self.collected = np.zeros(3, dtype=bool)
        self.steps = self.collisions = 0
        return self._observation(), self._info()

    def _clearance(self, angle):
        """Analytic ray clearance against inflated obstacles and world bounds."""
        x, y = self.position
        dx, dy = math.cos(angle), math.sin(angle)
        r = self.radius
        distances = []
        for value, direction, low, high in ((x, dx, r, self.width-r), (y, dy, r, self.height-r)):
            if abs(direction) > 1e-12:
                distances.append(((high if direction > 0 else low)-value)/direction)
        best = min(self.sensor_range, *distances)
        for ox, oy, w, h in self.obstacles:
            near, far = -float('inf'), float('inf')
            for value, direction, low, high in ((x, dx, ox-r, ox+w+r), (y, dy, oy-r, oy+h+r)):
                if abs(direction) < 1e-12:
                    if not low <= value <= high:
                        far = -float('inf')
                        break
                else:
                    a, b = (low-value)/direction, (high-value)/direction
                    near, far = max(near, min(a, b)), min(far, max(a, b))
            if far >= max(near, 0.):
                best = min(best, max(near, 0.))
        return max(0., best)/self.sensor_range

    def _observation(self):
        angles = [self.heading+.6, self.heading, self.heading-.6]
        remaining = self.foods[~self.collected]
        odors = []
        for angle in angles:
            antenna = self.position + .45*np.array([math.cos(angle), math.sin(angle)])
            distance = np.linalg.norm(remaining-antenna, axis=1).min() if len(remaining) else self.sensor_range
            odors.append(max(0., 1.-distance/self.sensor_range))
        return np.array(odors + [self._clearance(a) for a in angles] +
                        [math.sin(self.heading), math.cos(self.heading)], dtype=np.float32)

    def _info(self):
        return {'foods_collected': int(self.collected.sum()), 'collisions': self.collisions,
                'success': bool(self.collected.all()), 'steps': self.steps}

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError('Action must be 0 (left), 1 (forward), or 2 (right)')
        remaining = self.foods[~self.collected]
        before = np.linalg.norm(remaining-self.position, axis=1).min() if len(remaining) else 0.
        self.heading = (self.heading + (1-int(action))*self.turn + math.pi) % (2*math.pi)-math.pi
        proposed = self.position + self.speed*np.array([math.cos(self.heading), math.sin(self.heading)])
        collision = self._blocked(*proposed)
        if collision:
            self.collisions += 1
        else:
            self.position = proposed
        after = np.linalg.norm(remaining-self.position, axis=1).min() if len(remaining) else 0.
        newly = (~self.collected) & (np.linalg.norm(self.foods-self.position, axis=1) <= self.food_radius+self.radius)
        self.collected |= newly
        self.steps += 1
        reward = 10.*int(newly.sum()) + 2.*(before-after) - .01 - .15*collision
        success = bool(self.collected.all())
        return self._observation(), float(reward), success, self.steps >= self.horizon and not success, self._info()

    def snapshot(self):
        return {'environment': 'foraging', 'width': self.width, 'height': self.height,
                'fly': {'x': float(self.position[0]), 'y': float(self.position[1]),
                        'heading': self.heading, 'radius': self.radius},
                'foods': [{'x': float(p[0]), 'y': float(p[1]), 'radius': self.food_radius,
                           'collected': bool(c)} for p, c in zip(self.foods, self.collected)],
                'obstacles': [{'x': x, 'y': y, 'width': w, 'height': h} for x, y, w, h in self.obstacles],
                'step': self.steps, 'horizon': self.horizon,
                **self._info(), 'observation': self._observation().tolist()}


def make_env(environment='foraging', **kwargs):
    if environment != 'foraging':
        raise ValueError('Solo está disponible la simulación 2D de la mosca')
    return ForagingEnv(**kwargs)
