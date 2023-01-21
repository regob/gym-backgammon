import argparse
import time
from itertools import count
import random
import gymnasium as gym
import numpy as np
import torch

from gym_backgammon.envs.backgammon import WHITE, BLACK, COLORS, TOKEN
from agents import RandomAgent, LearningAgent, PubevalAgent, DumbevalAgent
from play_two_agents import play_two_agents


def train_agent(agent, n_games=1000, eval_agent=None, eval_freq=1000, eval_games=100, print_freq=50):

    t = time.time()
    total_loss, total_turns = 0.0, 0

    for game_idx in range(n_games):
        observation, info = env.reset()
        color = info["current_agent"]
        game_turns = 1

        while True:

            actions = env.get_valid_actions()
            action = agent.choose_best_action(actions, observation, env)

            observation_next, reward, terminated, truncated, info = env.step(action)

            if terminated:
                agent.step(observation, color, observation_next, reward)

            if terminated or truncated:
                break

            agent.step(observation, color, observation_next, None)
            observation = observation_next
            color = info["current_agent"]
            game_turns += 1

        total_loss += agent.total_loss
        total_turns += game_turns

        if print_freq and (game_idx + 1) % print_freq == 0:
            mean_loss = total_loss / max(total_turns, 1)
            norm = agent.weight_norm()
            print(f"Game {game_idx + 1:<6} done (avg turns: {total_turns//print_freq:<4}). Mean loss: {mean_loss:.8f}, weight norm: {norm:.6f}.")
            total_loss, total_turns = 0.0, 0

        # evaluate against the opponent agent if appropriate
        if eval_agent is not None and game_idx % eval_freq == eval_freq - 1:
            wins = play_two_agents(env, agent, eval_agent, eval_games)
            elapsed = time.time() - t
            exp_gain = (wins[0] - wins[1]) / sum(wins)
            print(f"After {game_idx + 1:<6} games done. Result against opponent: {wins[0]:<2}-{wins[1]:<2} ({exp_gain:.2f} gain). Elapsed {elapsed:.4f} s")

            # save temp state
            agent.save_state("temp_state.pth")

        agent.next_game()

    return agent


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("N", type=int)
    parser.add_argument("--lr", default=5e-4, type=float)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--maxlen", default=500, type=int)
    parser.add_argument("--eps", default=0.0, type=float)
    parser.add_argument("--load_state", action="store_true")
    parser.add_argument("--n_hidden", default=256, type=int)
    args = parser.parse_args()
    
    env = gym.make('gym_backgammon:backgammon-v0', render_mode="human")

    agent = LearningAgent(0,
                          env.observation_space.shape,
                          args.n_hidden,
                          lr=args.lr,
                          eps=args.eps,
                          batch_size=args.batch_size,
                          weight_decay=args.weight_decay,
                          debug=args.debug,
                          maxlen=args.maxlen)
    if args.load_state:
        agent.load_state()

    opponent_agent = PubevalAgent(1)

    N = args.N
    FREQ = 1000
    train_agent(agent, N, opponent_agent, FREQ, 300, 100)
    agent.save_state()
