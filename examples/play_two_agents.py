import gymnasium as gym
import time
from itertools import count
import random
import numpy as np

from gym_backgammon.envs.backgammon import WHITE, BLACK, COLORS, TOKEN
from agents import RandomAgent, LearningAgent, PubevalAgent, DumbevalAgent


def play_two_agents(env, agent1, agent2, n_games=1, verbose=False, render=False):
    """Make agent1 play n_games against agent2. Return (agent1_wins, agent2_wins). """
    
    agent_order = (agent1, agent2)
    wins = [0, 0]
    start_t = time.time()

    if verbose:
        print(f"========== {agent1.name} vs {agent2.name} ==========")

    def assign_agent_colors():
        if random.random() > 0.5:
            return {WHITE: agent_order[0], BLACK: agent_order[1]}
        return {WHITE: agent_order[1], BLACK: agent_order[0]}

    def agent_idx(a):
        return agent_order.index(a)

    for game_idx in range(n_games):
        agents_by_color = assign_agent_colors()
        observation, info = env.reset()
        finished = False
        num_turns = 1
        
        if render:
            env.render()

        
        while not finished:
            color = info["current_agent"]
            actions = env.get_valid_actions()
            action = agents_by_color[color].choose_best_action(actions, observation, env, training=False)
            observation, reward, terminated, truncated, info = env.step(action)

            if render:
                env.render()

            if terminated:
                winning_agent = agents_by_color[color]
                wins[agent_idx(winning_agent)] += 1

            if terminated or truncated:
                if verbose:
                    winner = "TRUNC" if truncated else agent_idx(agents_by_color[color])
                    n_games = max(sum(wins), 1)
                    msg = "Game={} | Winner={} || Turns={:<5} || Wins: {}={:<6}({:<5.1f}%) | {}={:<6}({:<5.1f}%) | Duration={:<.3f} sec"
                    print(msg.format(game_idx + 1, winner, num_turns,
                                     agent_order[0].name, wins[0], wins[0] / n_games * 100,
                                     agent_order[1].name, wins[1], wins[1] / n_games * 100, time.time() - start_t))
                break
            num_turns += 1
    return wins


if __name__ == '__main__':
    env = gym.make('gym_backgammon:backgammon-v0', render_mode="human")
    
    N = 3000
    agent1 = LearningAgent(0, env.observation_space.shape, lr=3e-4, eps=0.05, weight_decay=1e-4, debug=False)
    agent1.load_state()
    agent2 = PubevalAgent(1)
    play_two_agents(env, agent1, agent2, N, True, False)
