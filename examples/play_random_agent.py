import gymnasium as gym
import time
from itertools import count
import random
import numpy as np
from gym_backgammon.envs.backgammon import WHITE, BLACK, COLORS, TOKEN

env = gym.make('gym_backgammon:backgammon-v0', render_mode="human")
# env = gym.make('gym_backgammon:backgammon-pixel-v0')

random.seed(0)
np.random.seed(0)


class RandomAgent:
    def __init__(self, color):
        self.color = color
        self.name = 'AgentExample({})'.format(self.color)

    def choose_best_action(self, actions, env):
        return random.choice(list(actions)) if actions else None


def make_plays(n_games=1):
    wins = {WHITE: 0, BLACK: 0}

    agents = {WHITE: RandomAgent(WHITE), BLACK: RandomAgent(BLACK)}

    observation, info = env.reset()
    agent_color = info["current_agent"]
    agent = agents[agent_color]

    t = time.time()

    # env.render()

    game_idx = 1
    for i in count():
        roll = info["roll"]

        # print("Current player={} ({} - {}) | Roll={}".format(agent.color, TOKEN[agent.color], COLORS[agent.color], roll))
        actions = env.get_valid_actions()
        #print(actions)
        action = agent.choose_best_action(actions, env)

        observation_next, reward, terminated, truncated, info = env.step(action)
        #env.render()

        if terminated:
            wins[agent_color] += 1

            tot = wins[WHITE] + wins[BLACK]
            tot = tot if tot > 0 else 1

            print("Game={} | Winner={} after {:<4} plays || Wins: {}={:<6}({:<5.1f}%) | {}={:<6}({:<5.1f}%) | Duration={:<.3f} sec".format(1, agent_color, i,
                agents[WHITE].name, wins[WHITE], (wins[WHITE] / tot) * 100,
                agents[BLACK].name, wins[BLACK], (wins[BLACK] / tot) * 100, time.time() - t))
        if truncated:
            print("Game={} truncated".format(game_idx))

        if terminated or truncated:
            game_idx += 1
            if game_idx > n_games:
                break
            observation, info = env.reset()
        else:
            observation = observation_next
            
        agent_color = info["current_agent"]
        agent = agents[agent_color]
        
    env.close()


if __name__ == '__main__':
    start_t = time.time()
    N = 100
    make_plays(N)
    print(f"{N=} games took: {time.time() - start_t:.4f} s")
