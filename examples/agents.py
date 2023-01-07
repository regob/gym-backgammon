import random
import math
from collections import deque
import torch
import numpy as np
from torch import functional as F
from gym_backgammon.envs.backgammon_env import BackgammonEnv
from gym_backgammon.envs.backgammon import WHITE, BLACK


class RandomAgent:
    def __init__(self, idx):
        self.idx = idx
        self.name = 'RandomAgent({})'.format(self.idx)

    def choose_best_action(self, actions, observation, env):
        return random.choice(list(actions)) if actions else None

    def reward(self, r=None):
        pass

    def next_game(self, color):
        self.color = color


class LearningAgent:
    def __init__(self, idx, observation_shape, lr=0.05, eps=0.1, maxlen=100, batch_size=8, gamma=0.7):
        self.idx = idx
        self.observation_shape = observation_shape
        self.name = 'LearningAgent({})'.format(self.idx)
        self.eps = eps
        self.lr = lr

        # input layer neurons
        self.n_input = math.prod(self.observation_shape)
        # output layer neurons
        self.n_out = 1
        # hidden layer neurons
        self.n_hidden = 300

        self.layer1 = torch.nn.Linear(self.n_input, self.n_hidden)
        self.layer2 = torch.nn.Linear(self.n_hidden, self.n_hidden)
        self.layer3 = torch.nn.Linear(self.n_hidden, self.n_out)

        torch.nn.init.xavier_normal_(self.layer1.weight)
        torch.nn.init.xavier_normal_(self.layer3.weight)

        self.net = torch.nn.Sequential(
            self.layer1, torch.nn.ReLU(),
            self.layer2, torch.nn.ReLU(),
            self.layer3, torch.nn.Sigmoid(),
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.net.to(self.device)
        print(f"{self.n_input=}, {self.n_out=}")

        self.maxlen = maxlen
        self.observations = deque([], maxlen=self.maxlen)
        self.saved_states = deque([], maxlen=self.maxlen)
        self.scores = deque([], maxlen=self.maxlen)
        self.optim = torch.optim.AdamW(self.net.parameters(), lr=self.lr, amsgrad=True)
        self.optim.zero_grad()

        self.batch_size = batch_size
        self.gamma = gamma

    def next_game(self, color):
        self.color = color

    def choose_best_action(self, actions, observation, env: BackgammonEnv):
        saved_state = env.game.save_state()
        observation = torch.Tensor(observation)

        is_white = self.color == WHITE

        best_action, best_score = None, -1e9
        if actions:
            if random.random() > self.eps:
                return random.choice(list(actions))
            
            for action in actions:
                internal_action = tuple((env.internal_action_map[x], env.internal_action_map[y])
                                        for x, y in action)

                env.game.execute_play(env.current_agent, internal_action)
                next_observation = torch.Tensor(env.game.get_board_features(env.game.get_opponent(env.current_agent)))
                with torch.no_grad():
                    score = self.net(next_observation.reshape(1, -1).to(self.device)).item()
                    if not is_white:
                        score = 1.0 - score
                if score > best_score:
                    best_action, best_score = action, score
                env.game.restore_state(saved_state)

        with torch.no_grad():
            score = self.net(observation.reshape(1, -1).to(self.device)).item()
        if not is_white:
            score = 1.0 - score

        self.scores.append(score)
        self.saved_states.append(saved_state)
        self.observations.append(observation)

        return best_action

    def reward(self, r=None):
        if len(self.observations) < 2:
            return
        # if r is None:
        #     return
        # next_score = r
        # i = 0
        # for observation in reversed(self.observations):
        #     score = self.net(observation.reshape(1, -1).to(self.device)).reshape(-1)
        #     if self.color != WHITE:
        #         score = 1.0 - score
        #     loss = torch.abs(next_score - score)
        #     if i == 0:
        #         i += 1
        #         # print(score, r)
            
        #     self.optim.zero_grad()
        #     loss.backward()
        #     self.optim.step()
        #     next_score = score.detach()

        for p in self.net.parameters():
            p.grad *= self.gamma
            
        score = self.net(self.observations[-2].reshape(1, -1)).reshape(-1)
        if self.color != WHITE:
            score = 1.0 - score
            
        if r is not None:
            loss = torch.abs(r - score)
        else:
            loss = torch.abs(self.scores[-1] - score)
        loss.backward()
        
        self.optim.step()
        # self.optim.zero_grad()

    def save_state(self, path="learning_agent_state.pth"):
        torch.save(self.net.state_dict(), path)

    def load_state(self, path="learning_agent_state.pth"):
        state_dict = torch.load(path)
        self.net.load_state_dict(state_dict)
