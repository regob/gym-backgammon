import sys
import random
import math
from collections import deque
import torch
import numpy as np
from torch import functional as F
from gym_backgammon.envs.backgammon_env import BackgammonEnv
from gym_backgammon.envs.backgammon import WHITE, BLACK

sys.path.append(__file__)
from pubeval import pubeval
from dumbeval import dumbeval


class RandomAgent:
    def __init__(self, idx):
        self.idx = idx
        self.name = 'RandomAgent({})'.format(self.idx)

    def choose_best_action(self, actions, observation, env, training=False):
        return random.choice(list(actions)) if actions else None

    def next_game(self):
        pass

class PolicyAgent:
    def __init__(self, idx, policy_fn):
        self.policy_fn = policy_fn

    def choose_best_action(self, actions, observation, env, training=False):
        if len(actions) == 0:
            return None
        best_action, best_score = None, -1e9
        saved_state = env.game.save_state()
        
        for action in actions:
            internal_action = tuple((env.internal_action_map[x], env.internal_action_map[y])
                                        for x, y in action)
            env.game.execute_play(env.current_agent, internal_action)
            
            # get the board after the move from the current player's perspective
            board = env.game.get_board_pubeval(env.current_agent)
            score = self.policy_fn(board)
            
            if score > best_score:
                best_action, best_score = action, score
            
            env.game.restore_state(saved_state)

        return best_action

class DumbevalAgent(PolicyAgent):
    def __init__(self, idx):
        super().__init__(idx, dumbeval)
        self.idx = idx
        self.name = "DumbevalAgent({})".format(self.idx)

class PubevalAgent(PolicyAgent):
    def __init__(self, idx):
        super().__init__(idx, pubeval)
        self.idx = idx
        self.name = "PubevalAgent({})".format(self.idx)
        

class LearningAgent:
    def __init__(self, idx, observation_shape, lr=0.05, eps=0.1, weight_decay=1e-6, batch_size=16, debug=False,
                 debug_freq=100, maxlen=150):
        self.idx = idx
        self.observation_shape = observation_shape
        self.name = 'LearningAgent({})'.format(self.idx)
        self.eps = eps
        self.lr = lr
        self.batch_size = batch_size

        # input layer neurons
        self.n_input = math.prod(self.observation_shape)
        # output layer neurons (p_white wins, p_black wins)
        self.n_out = 2
        # hidden layer neurons
        self.n_hidden = 256

        self.layer1 = torch.nn.Linear(self.n_input, self.n_hidden)
        # self.layer2 = torch.nn.Linear(self.n_hidden, self.n_hidden)
        self.layer3 = torch.nn.Linear(self.n_hidden, self.n_out)

        torch.nn.init.xavier_normal_(self.layer1.weight)
        torch.nn.init.xavier_normal_(self.layer3.weight)

        self.net = torch.nn.Sequential(
            self.layer1, torch.nn.ReLU(),
            #            self.layer2, torch.nn.ReLU(),
            self.layer3, torch.nn.Sigmoid(),
        )
        self.net.train()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.net.to(self.device)
        print("LearningAgent uses network:", self.net)

        self.optim = torch.optim.SGD(self.net.parameters(), lr=self.lr, momentum=0.9, weight_decay=weight_decay)
        self.optim.zero_grad()
        self.loss = torch.nn.MSELoss()

        self.total_loss = 0.0
        self.num_turns = 0
        self.num_rounds = 1
        self.exploration_move = False
        self.debug = debug
        self.debug_freq = debug_freq

        self.playback_buffer = deque([], maxlen=maxlen)

    def _set_optim_par(self, name, value):
        self.optim.param_groups[0][name] = value

    def _set_lr(self, lr):
        self.lr = lr
        self._set_optim_par("lr", lr)

    def next_game(self):
        self.total_loss, self.num_turns = 0.0, 0
        self.num_rounds += 1

    def weight_norm(self):
        norm = 0.0
        for p in self.net.parameters():
            norm += p.detach().data.norm(2).item() ** 2
        return norm ** 0.5

    def choose_best_action(self, actions, observation, env: BackgammonEnv, training=True):
        saved_state = env.game.save_state()
        observation = torch.Tensor(observation)

        is_white = env.current_agent == WHITE
        player_idx = 0 if is_white else 1

        best_action = None
        self.exploration_move = False
        
        self.net.eval()

        if actions:
            if training and random.random() < self.eps:
                self.exploration_move = True
                return random.choice(list(actions))

            next_observations = []
            for action in actions:
                internal_action = tuple((env.internal_action_map[x], env.internal_action_map[y])
                                        for x, y in action)

                env.game.execute_play(env.current_agent, internal_action)
                next_observation = torch.Tensor(env.game.get_board_features(env.game.get_opponent(env.current_agent)))
                next_observations.append(next_observation)
                env.game.restore_state(saved_state)

            next_observations = torch.stack(next_observations)
            with torch.no_grad():
                scores = self.net(next_observations.to(self.device))[:, player_idx].reshape(-1)

                best_action_idx = scores.argmax()
                best_action = actions[best_action_idx]

        if self.num_rounds % self.debug_freq == 0 and training and self.debug:
            with torch.no_grad():
                score = self.net(observation.reshape(1, -1).to(self.device))[0]
            print("Agent", "WHITE" if is_white else "BLACK", score)

        return best_action

    def step(self, observation, color, observation_next, reward=None):

        if self.exploration_move:
            return

        self.net.train()

        observation_tensor = torch.Tensor(observation).reshape(1, -1).to(self.device)
        observation_next_tensor = torch.Tensor(observation_next).reshape(1, -1).to(self.device)

        round_end = reward is not None
        if reward is None:
            with torch.no_grad():
                reward = self.net(observation_next_tensor)
        else:
            reward = torch.FloatTensor([reward, 1.0 - reward]).reshape(1, -1).to(self.device)
        self.playback_buffer.append((observation_tensor, reward))

        if self.batch_size > 1:
            sample = random.sample(self.playback_buffer, min(self.batch_size - 1, len(self.playback_buffer)))
        else:
            sample = []
        sample.append((observation_tensor, reward))
            
        X = torch.cat([x[0] for x in sample]).to(self.device)
        Y = torch.cat([x[1] for x in sample]).to(self.device)
        
        prev_scores = self.net(X)

        loss = self.loss(prev_scores, Y)
        loss.backward()
        self.optim.step()
        self.optim.zero_grad()

        if round_end and self.debug:
            with torch.no_grad():
                better_score = self.net(observation_tensor).reshape(-1)

            pscores = list(prev_scores.detach().cpu().numpy())[-1]
            print(f"[{pscores[0]:.8f}, {pscores[1]:.8f}]", end=" ")
            bscore = list(better_score.cpu().numpy())
            print(f"[{bscore[0]:.8f}, {bscore[1]:.8f}]", end=" ")
            rew = list(reward[0].cpu().numpy())
            print(f"[{int(rew[0])}, {int(rew[1])}]", end=" ")
            w = self.net[0].weight[0][0].item()
            print(f"{w:.8f}")


        self.total_loss += loss.item()
        self.num_turns += 1

        # self.num_turns += 1

        # score = self.net(self.observations[-2].reshape(1, -1).to(self.device)).reshape(-1)
        # if self.color != WHITE:
        #     score = 1.0 - score

        # if r is not None:
        #     loss = torch.abs(r - score)
        #     self.total_loss += loss.item()
        #     # print(f"Average loss ({self.num_turns} turns): {self.total_loss / self.num_turns}")
        #     self.total_loss = 0.0
        #     self.num_turns = 0
        #     self.num_rounds += 1
        # else:
        #     loss = torch.abs(self.scores[-1] - score)
        #     self.total_loss += loss.item()
        # loss.backward()

        # self.optim.step()
        # self.optim.zero_grad()

    def save_state(self, path="learning_agent_state.pth"):
        torch.save(self.net.cpu().state_dict(), path)
        self.net.to(self.device)

    def load_state(self, path="learning_agent_state.pth"):
        state_dict = torch.load(path)
        self.net.load_state_dict(state_dict)
