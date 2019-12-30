import numpy as np


class DynaMaze:
    def __init__(self, mode):
        self._mode = mode
        self._board_width = 9
        self._board_height = 6
        self._dir = np.array([[-1, 0], [1, 0], [0, 1], [0, -1]])
        self.road = 0
        self.player = 1
        self.goal = 2
        self.obstacle = 3
        self.cnt_step = 0
        if mode == "initial":
            self._start = np.array([2, 0])
            self._goal = np.array([0, 8])
            self._obstacle = [[1, 2], [2, 2], [3, 2], [4, 5], [0, 7], [1, 7], [2, 7]]
        self.maze = self.create_maze()
        self.cur_pos = self._start

    def create_maze(self):
        maze = [[0 for i in range(self._board_width)] for i in range(self._board_height)]
        maze[self._start[0]][self._start[1]] = self.player
        maze[self._goal[0]][self._goal[1]] = self.goal
        for obt in self._obstacle:
            maze[obt[0]][obt[1]] = self.obstacle
        return maze

    def reset(self):
        self.cur_pos = self._start
        self.maze = self.create_maze()
        self.cnt_step = 0

    def check_pos(self, pos):
        if pos[0] < 0 or pos[0] >= self._board_height or pos[1] < 0 or pos[1] >= self._board_width or \
                self.maze[pos[0]][pos[1]] == 3:
            return False
        return True

    def step(self, action):
        nxt_pos = self.cur_pos + self._dir[action]
        self.cnt_step += 1
        reward = 0
        terminal = 0
        if self.check_pos(nxt_pos):
            if self.maze[nxt_pos[0]][nxt_pos[1]] == self.goal:
                reward = 1
                terminal = 1
            self.maze[self.cur_pos[0]][self.cur_pos[1]] = self.road
            self.maze[nxt_pos[0]][nxt_pos[1]] = self.player
            self.cur_pos = nxt_pos
        return self.maze, reward, terminal, self.cnt_step


