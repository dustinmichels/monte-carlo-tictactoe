"""
Test out different AI agents against each other.

Adapted from:
https://github.com/haje01/gym-tictactoe/blob/master/examples/base_agent.py
"""

import random
from collections import Counter
from copy import copy

from gym_tictactoe.env import TicTacToeEnv, agent_by_mark, next_mark
from tqdm import tqdm, trange

from agents import TicTacJoe, TicTacPro

players = {"X": TicTacJoe, "O": TicTacPro}


def play(num_games, verbose=True):
    """
    Test out two agents playing against each other.
    Displays progress and result.

    Parameters:
    -----------
    num_games: int
        How many games to simulate
    verbose: bool
        If true, display play information during each game
        If false, just display progress bar as simulations progress.
    """
    # Print header
    print("-" * 30)
    print(f"Playing {num_games} games")
    print("  * Player X: {}".format(players["X"].name))
    print("  * Player O: {}".format(players["O"].name))
    print("-" * 30)

    # select random starting player
    start_mark = random.choice(["X", "O"])

    # keep track of who won
    winners = []

    # if verbose is false, display progress bar
    if not verbose:
        myrange = trange
    else:
        myrange = range

    for _ in myrange(num_games):

        # set up board
        env = TicTacToeEnv()
        env.set_start_mark(start_mark)
        state = env.reset()

        # init the agents
        agents = [players["X"]("X"), players["O"]("O")]

        # play until game is done
        while not env.done:
            _, mark = state
            if verbose:
                env.show_turn(True, mark)
            agent = agent_by_mark(agents, mark)
            action = agent.act(state, copy(env))
            state, reward, _, _ = env.step(action)
            if verbose:
                env.render()

        # append winner to list (-1=X, 1=0, 0=tie)
        winners.append(reward)

        # print out result
        if verbose:
            env.show_result(True, mark, reward)

        # rotate start
        start_mark = next_mark(start_mark)

    # tally and display final stats
    c = Counter(winners)
    total = c[-1] + c[1] + c[0]
    print("\nX won {} ({:.2%})".format(c[-1], c[-1] / total))
    print("O won {} ({:.2%})".format(c[1], c[1] / total))
    print("Tied  {} ({:.2%})".format(c[0], c[0] / total))


if __name__ == "__main__":
    play(num_games=5)
    # play(num_games=100, verbose=False)
