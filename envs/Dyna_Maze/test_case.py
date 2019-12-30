import unittest
import numpy as np
from envs.Dyna_Maze.env import DynaMaze


class TestFunction(unittest.TestCase):

    def test_mode_initial(self):
        env = DynaMaze("initial")
        env.reset()
        maze = [[0, 0, 0, 0, 0, 0, 0, 3, 2],
                [0, 0, 3, 0, 0, 0, 0, 3, 0],
                [1, 0, 3, 0, 0, 0, 0, 3, 0],
                [0, 0, 3, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 3, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0]]
        # Also test the create_maze function.
        self.assertEqual(env.maze, maze)

    def test_reset(self):
        env = DynaMaze("initial")
        env.reset()
        start = [2, 0]
        self.assertEqual(env.cur_pos.tolist(), start)
        self.assertEqual(env.cnt_step, 0)

    def test_check_pos(self):
        env = DynaMaze("initial")

        # Out of bounds
        self.assertEqual(env.check_pos([-1, 0]), False)
        self.assertEqual(env.check_pos([6, 0]), False)
        self.assertEqual(env.check_pos([0, -1]), False)
        self.assertEqual(env.check_pos([0, 9]), False)
        # Obstacle

        self.assertEqual(env.check_pos([1, 2]), False)

        # Road
        self.assertEqual(env.check_pos([0, 0]), True)
        self.assertEqual(env.check_pos([5, 8]), True)

    def test_step(self):
        env = DynaMaze("initial")
        env.reset()
        env.maze[2][0] = 0
        # Blocked by obstacle
        pos1 = [1, 3]
        env.cur_pos = np.array(pos1)
        env.maze[env.cur_pos[0]][env.cur_pos[1]] = 1
        obs, reward, done, cnt_step = env.step(3)
        maze = [[0, 0, 0, 0, 0, 0, 0, 3, 2],
                [0, 0, 3, 1, 0, 0, 0, 3, 0],
                [0, 0, 3, 0, 0, 0, 0, 3, 0],
                [0, 0, 3, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 3, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0]]
        self.assertEqual(env.cur_pos.tolist(), pos1)
        self.assertEqual(obs, maze)
        self.assertEqual(reward, 0)
        self.assertEqual(done, 0)
        self.assertEqual(cnt_step, 1)
        env.maze[env.cur_pos[0]][env.cur_pos[1]] = 0
        pos2 = [3, 7]
        env.cur_pos = np.array(pos2)
        env.maze[env.cur_pos[0]][env.cur_pos[1]] = 1
        obs, reward, done, _ = env.step(0)
        maze = [[0, 0, 0, 0, 0, 0, 0, 3, 2],
                [0, 0, 3, 0, 0, 0, 0, 3, 0],
                [0, 0, 3, 0, 0, 0, 0, 3, 0],
                [0, 0, 3, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 3, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0]]
        self.assertEqual(env.cur_pos.tolist(), pos2)
        self.assertEqual(obs, maze)
        self.assertEqual(reward, 0)
        self.assertEqual(done, 0)
        env.maze[env.cur_pos[0]][env.cur_pos[1]] = 0

        # Out of bounds
        pos1 = [0, 0]
        env.cur_pos = np.array(pos1)
        env.maze[env.cur_pos[0]][env.cur_pos[1]] = 1
        obs, reward, done, _ = env.step(0)
        maze = [[1, 0, 0, 0, 0, 0, 0, 3, 2],
                [0, 0, 3, 0, 0, 0, 0, 3, 0],
                [0, 0, 3, 0, 0, 0, 0, 3, 0],
                [0, 0, 3, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 3, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0]]
        self.assertEqual(env.cur_pos.tolist(), pos1)
        self.assertEqual(obs, maze)
        self.assertEqual(reward, 0)
        self.assertEqual(done, 0)
        obs, reward, done, _ = env.step(3)
        self.assertEqual(env.cur_pos.tolist(), pos1)
        self.assertEqual(obs, maze)
        self.assertEqual(reward, 0)
        self.assertEqual(done, 0)
        env.maze[env.cur_pos[0]][env.cur_pos[1]] = 0

        # Walk on the road
        pos = [2, 4]
        env.cur_pos = np.array(pos)
        env.maze[env.cur_pos[0]][env.cur_pos[1]] = 1
        obs, reward, done, _ = env.step(1)
        maze = [[0, 0, 0, 0, 0, 0, 0, 3, 2],
                [0, 0, 3, 0, 0, 0, 0, 3, 0],
                [0, 0, 3, 0, 0, 0, 0, 3, 0],
                [0, 0, 3, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 3, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0]]
        self.assertEqual(env.cur_pos.tolist(), [3, 4])
        self.assertEqual(obs, maze)
        self.assertEqual(reward, 0)
        self.assertEqual(done, 0)
        env.maze[env.cur_pos[0]][env.cur_pos[1]] = 0

        # Reach the goal
        pos = [1, 8]
        env.cur_pos = np.array(pos)
        env.maze[env.cur_pos[0]][env.cur_pos[1]] = 1
        obs, reward, done, _ = env.step(0)
        maze = [[0, 0, 0, 0, 0, 0, 0, 3, 1],
                [0, 0, 3, 0, 0, 0, 0, 3, 0],
                [0, 0, 3, 0, 0, 0, 0, 3, 0],
                [0, 0, 3, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 3, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0]]
        self.assertEqual(env.cur_pos.tolist(), [0, 8])
        self.assertEqual(obs, maze)
        self.assertEqual(reward, 1)
        self.assertEqual(done, 1)
        env.maze[env.cur_pos[0]][env.cur_pos[1]] = 0
