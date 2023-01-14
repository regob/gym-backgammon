import gymnasium as gym
from gymnasium.spaces import Box, Sequence, Tuple, Discrete
from gym_backgammon.envs.backgammon import Backgammon as Game, WHITE, BLACK, COLORS, BAR, OFF
from random import randint
from gym_backgammon.envs.rendering import Viewer
import numpy as np

STATE_W = 96
STATE_H = 96

SCREEN_W = 600
SCREEN_H = 500


class BackgammonEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array', 'state_pixels']}

    def __init__(self, render_mode=None, max_length=1000):
        self.game = Game()
        self.roll = None
        self.current_agent = None

        low = np.zeros((198, 1))
        high = np.ones((198, 1))

        for i in range(3, 97, 4):
            high[i] = 6.0
        high[96] = 7.5

        for i in range(101, 195, 4):
            high[i] = 6.0
        high[194] = 7.5

        self.observation_space = Box(low=low, high=high, dtype= np.float32)
        # actions: [0-23] board fields, -1 and 24: off, 25:bar
        self.action_space = Sequence(Tuple([Discrete(27, start=-1), Discrete(27, start=-1)]))
        self.action_map = {i: i for i in range(-1, 25)}
        self.action_map[BAR] = 25
        self.internal_action_map = {v: k for k, v in self.action_map.items()}

        self.counter = 0
        self.max_length = max_length
        self.viewer = None
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def roll_dice(self):
        if self.current_agent == WHITE:
            return (-self.np_random.integers(1, 7), -self.np_random.integers(1, 7))
        return (self.np_random.integers(1, 7), self.np_random.integers(1, 7))

    def step(self, action):
        if action is not None:
            internal_action = tuple((self.internal_action_map[x], self.internal_action_map[y])
                                    for x, y in action)
        else:
            internal_action = None
        self.game.execute_play(self.current_agent, internal_action)

        # get the board representation from the opponent player perspective (the current player has already performed the move)
        observation = np.array(self.game.get_board_features(self.game.get_opponent(self.current_agent)), dtype=np.float32)
        
        reward = 0
        done = False

        winner = self.game.get_winner()

        if winner is not None:
            if winner == WHITE:
                reward = 1
            done = True

        self.counter += 1
        truncated = self.counter == self.max_length
        if not truncated and not done:
            self.current_agent = self.get_opponent_agent()
            self.roll = self.roll_dice()

        info = {"roll": self.roll, "current_agent": self.current_agent}
        return observation, reward, done, truncated, info

    def reset(self, seed=None):
        super().reset(seed=seed)

        # roll the dice
        roll = self.np_random.integers(1, 7), self.np_random.integers(1, 7)

        # roll the dice until they are different
        while roll[0] == roll[1]:
            roll = self.np_random.integers(1, 7), self.np_random.integers(1, 7)

        # set the current agent
        if roll[0] > roll[1]:
            self.current_agent = WHITE
            roll = (-roll[0], -roll[1])
        else:
            self.current_agent = BLACK

        self.game = Game()
        self.counter = 0

        self.roll = roll
        info = {"roll": self.roll, "current_agent": self.current_agent}

        return np.array(self.game.get_board_features(self.current_agent)), info

    def render(self):
        if self.render_mode == 'human':
            self.game.render()
            return True
        else:
            if self.viewer is None:
                self.viewer = Viewer(SCREEN_W, SCREEN_H)

            if self.render_mode == 'rgb_array':
                width = SCREEN_W
                height = SCREEN_H

            else:
                width = STATE_W
                height = STATE_H

        self.viewer.render(board=self.game.board, bar=self.game.bar, off=self.game.off, state_w=width, state_h=height)

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def get_valid_actions(self):
        internal_actions = self.game.get_valid_plays(self.current_agent, self.roll)
        if internal_actions is None:
            return ()
        return tuple(tuple((self.action_map[x], self.action_map[y]) for x, y in action)
                     for action in internal_actions)

    def get_opponent_agent(self):
        self.current_agent = self.game.get_opponent(self.current_agent)
        return self.current_agent


class BackgammonEnvPixel(BackgammonEnv):

    def __init__(self):
        super(BackgammonEnvPixel, self).__init__()
        self.observation_space = Box(low=0, high=255, shape=(STATE_H, STATE_W, 3), dtype=np.uint8)

    def step(self, action):
        observation, reward, done, winner = super().step(action)
        observation = self.render()
        return observation, reward, done, winner

    def reset(self):
        current_agent, roll, observation = super().reset()
        observation = self.render()
        return current_agent, roll, observation
