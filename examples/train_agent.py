import gymnasium as gym
import time
from itertools import count
import random
import numpy as np

from gym_backgammon.envs.backgammon import WHITE, BLACK, COLORS, TOKEN
from agents import RandomAgent, LearningAgent

def assign_agent_sides(agent1, agent2):
    if random.random() > 0.5:
        d = {WHITE: agent1, BLACK: agent2}
    else:
        d = {WHITE: agent2, BLACK: agent1}
    for color, agent in d.items():
        agent.next_game(color)
    return d

def make_plays(agent1, agent2, n_games=1, print_freq=50):
    agent_order = (agent1, agent2)
    wins = [0, 0]
    agent_idx = lambda ag: agent_order.index(ag)
    
    agents = assign_agent_sides(*agent_order)

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
        action = agent.choose_best_action(actions, observation, env)

        observation_next, reward, terminated, truncated, info = env.step(action)
        #env.render()

        if terminated:
            wins[agent_idx(agent)] += 1
            opponent_color = WHITE if agent_color == BLACK else BLACK
            agents[agent_color].reward(reward)
            agents[opponent_color].reward(1-reward)

            tot = wins[WHITE] + wins[BLACK]
            tot = tot if tot > 0 else 1

            if game_idx % print_freq == 0:
                print("Game={} | Winner={} after {:<4} plays || Wins: {}={:<6}({:<5.1f}%) | {}={:<6}({:<5.1f}%) | Duration={:<.3f} sec".format(game_idx, agent.idx, i,
                agent_order[0].name, wins[0], (wins[0] / tot) * 100,
                agent_order[1].name, wins[1], (wins[1] / tot) * 100, time.time() - t))
        if truncated:
            print("Game={} truncated".format(game_idx))

        if terminated or truncated:
            game_idx += 1
            if game_idx > n_games:
                break
            agents = assign_agent_sides(*agent_order)
            observation, info = env.reset()
        else:
            observation = observation_next
            agents[agent_color].reward()

        agent_color = info["current_agent"]
        agent = agents[agent_color]
        
    env.close()


if __name__ == '__main__':
    env = gym.make('gym_backgammon:backgammon-v0', render_mode="human")
    # env = gym.make('gym_backgammon:backgammon-pixel-v0')
    print(env.observation_space.shape)
    # random.seed(0)
    # np.random.seed(0)

    start_t = time.time()
    N = 10000
    agent1 = RandomAgent(0)
    agent2 = LearningAgent(1, env.observation_space.shape, lr=0.00005, eps=0.1, maxlen=300, gamma=0.7, init_rounds=100)
#    agent2.load_state()
    make_plays(agent1, agent2, N, 1)
    agent2.save_state()
    print(f"{N=} games took: {time.time() - start_t:.4f} s")
