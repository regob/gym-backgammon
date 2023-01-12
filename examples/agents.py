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

MX = 220
win_prob = np.ones((MX, MX), dtype=float)
for i in range(MX):
    for j in range(i, MX):
        # compute [i][j]
        total = 0.0
        for k in range(36):
            n1, n2 = k // 6 + 1, k % 6 + 1
            col = i - n1 - n2 if n1 != n2 else i - (n1 + n2) * 2
            if col <= 0:
                total += 1.0
            else:
                total += 1.0 - win_prob[j, col]
        win_prob[i, j] = total / 36

        # compute [j, i]
        total = 0.0
        for k in range(36):
            n1, n2 = k // 6 + 1, k % 6 + 1
            col = j - n1 - n2 if n1 != n2 else j - (n1 + n2) * 2
            if col <= 0:
                total += 1.0
            else:
                total += 1.0 - win_prob[i, col]
        win_prob[j, i] = total / 36
# print(f"{win_prob[150, 200]=}")
# print(f"{win_prob[219, 219]=}")
# print(f"{win_prob[70, 36]=}")

        
class LearningAgent:
    def __init__(self, idx, observation_shape, lr=0.05, eps=0.1, maxlen=100, batch_size=8, gamma=0.7, init_rounds=100):
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
        self.n_hidden = 500

        self.layer1 = torch.nn.Linear(self.n_input, self.n_hidden)
        self.layer2 = torch.nn.Linear(self.n_hidden, self.n_hidden)
        self.layer3 = torch.nn.Linear(self.n_hidden, self.n_out)

        torch.nn.init.xavier_normal_(self.layer1.weight)
        torch.nn.init.xavier_normal_(self.layer3.weight)

        self.net = torch.nn.Sequential(
            self.layer1, torch.nn.ReLU(),
#            self.layer2, torch.nn.ReLU(),
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
        self.init_rounds = init_rounds

        self.total_loss = 0.0
        self.num_turns = 0
        self.num_rounds = 0



    def next_game(self, color):
        self.color = color

    def choose_best_action(self, actions, observation, env: BackgammonEnv):
        saved_state = env.game.save_state()
        observation = torch.Tensor(observation)

        is_white = self.color == WHITE

        best_action, best_score = None, -1e9
        if actions:
            if random.random() < self.eps:
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

        if self.num_rounds < self.init_rounds:
            observation = self.observations[-1].reshape(-1)
            
            dist_white = 0.0
            for i in range(96):
                if i % 4 == 3:
                    cnt = observation[i] * 2.0
                else:
                    cnt = observation[i]
                dist_white += cnt * (24 - i // 4)

            dist_white += observation[96] * 2.0 * 25.0

            dist_black = 0.0
            for i in range(98, 194):
                if i % 4 == 1:
                    cnt = observation[i] * 2.0
                else:
                    cnt = observation[i]
                dist_black += cnt * (24 - (i - 98) // 4)

            dist_black += observation[194] * 2.0 * 25.0

            dist_white = min(int(dist_white), MX - 1)
            dist_black = min(int(dist_black), MX - 1)
            p_white = win_prob[dist_white, dist_black] if observation[196] > 0.5 else 1 - win_prob[dist_black, dist_white]
            score = self.net(observation.reshape(1, -1).to(self.device)).reshape(-1)
            loss = torch.abs(score - p_white)
            self.total_loss += loss.item()
            self.num_turns += 1
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            if r is not None:
                print(f"Average loss ({self.num_turns} turns): {self.total_loss / self.num_turns}")
                self.total_loss = 0
                self.num_turns = 0
                self.num_rounds += 1

            return

        if r is None:
            return
        next_score = r
        i = 0
        for observation in reversed(self.observations):
            score = self.net(observation.reshape(1, -1).to(self.device)).reshape(-1)
            if self.color != WHITE:
                score = 1.0 - score
            loss = torch.abs(next_score - score)
            if i == 0:
                i += 1
                print(score, r)

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            with torch.no_grad():
                next_score = self.net(observation.reshape(1, -1).to(self.device)).reshape(-1)




        # for p in self.net.parameters():
        #     if p.grad is not None:
        #         p.grad *= self.gamma

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

    def load_state(self, path="learning_agent_state.pth"):
        state_dict = torch.load(path)
        self.net.load_state_dict(state_dict)
