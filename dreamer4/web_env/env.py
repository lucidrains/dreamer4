import numpy as np

class SnakeEnv:
    def __init__(
        self,
        *,
        grid_size = 8,
        max_steps = 40
    ):
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.cell_size = 64 // grid_size
        self.action_space = 4
        self.reset()

    def reset(self, **kwargs):
        self.steps = 0
        self.snake = [(self.grid_size // 2, self.grid_size // 2)]
        self.direction = 1
        self.food = self._place_food()
        self.done = False
        return dict(image = self._render())

    def _place_food(self):
        if len(self.snake) >= self.grid_size ** 2:
            return (-1, -1)

        while True:
            food = (int(np.random.randint(self.grid_size)), int(np.random.randint(self.grid_size)))
            if food not in self.snake:
                return food

    def step(self, action):
        if self.done:
            return dict(image = self._render()), 0.0, True, False, {}

        self.steps += 1

        if isinstance(action, np.ndarray) or hasattr(action, 'item'):
            action = action.item()

        action = int(action)

        if abs(action - self.direction) != 2:
            self.direction = action

        dx, dy = ((0, -1), (1, 0), (0, 1), (-1, 0))[self.direction]
        hx, hy = self.snake[0]
        new_head = (hx + dx, hy + dy)

        reward = 0.0
        terminated = False
        truncated = self.steps >= self.max_steps

        if (
            new_head[0] < 0 or new_head[0] >= self.grid_size or
            new_head[1] < 0 or new_head[1] >= self.grid_size or
            new_head in self.snake
        ):
            terminated = True
            reward = -1.0
        else:
            self.snake.insert(0, new_head)
            if new_head == self.food:
                reward = 1.0
                self.food = self._place_food()
            else:
                self.snake.pop()

        self.done = terminated or truncated
        return dict(image = self._render()), reward, terminated, truncated, {}

    def _render(self):
        img = np.zeros((3, 64, 64), dtype = np.float32)
        c = self.cell_size

        fx, fy = self.food
        img[0, fy*c:(fy+1)*c, fx*c:(fx+1)*c] = 1.0

        for i, (sx, sy) in enumerate(self.snake):
            y1, y2, x1, x2 = sy*c, (sy+1)*c, sx*c, (sx+1)*c
            img[1, y1:y2, x1:x2] = 1.0 if i == 0 else 0.8

            if i == 0:
                if   self.direction == 0: slice_y, slice_x = slice(y1, y1+1), slice(x1, x2)
                elif self.direction == 1: slice_y, slice_x = slice(y1, y2), slice(x2-1, x2)
                elif self.direction == 2: slice_y, slice_x = slice(y2-1, y2), slice(x1, x2)
                elif self.direction == 3: slice_y, slice_x = slice(y1, y2), slice(x1, x1+1)

                img[:, slice_y, slice_x] = 1.0

        return img
