import math
import random
import typing
from copy import deepcopy
from typing import Generic, List, Optional, TypeVar, Literal

from gym_tictactoe.env import (
    TicTacToeEnv,
    after_action_state,
    agent_by_mark,
    check_game_status,
    next_mark,
    tocode,
    tomark,
)

NODE_ID_COUNTER = 0

Mark = Literal["X", "O"]


class BaseAgent:
    """
    Most basic agent.
    Plays winning move if possible, random otherwise.
    """

    name = "BaseAgent"

    def __init__(self, mark: Mark):
        self.mark = mark

    @property
    def opponent_mark(self):
        return next_mark(self.mark)

    def act(self, state, my_env):
        available_actions = my_env.available_actions()
        for action in available_actions:
            nstate = after_action_state(my_env.state, action)
            gstatus = check_game_status(nstate[0])
            if gstatus > 0:
                if tomark(gstatus) == self.mark:
                    return action
        return random.choice(available_actions)


class TicTacJoe(BaseAgent):
    """
    A slightly more interesting agent.
    Tries to:
        1. Play a winning move
        2. Prevent opponent from winning
    """

    name = "TicTacJoe"

    def act(self, state, my_env: TicTacToeEnv):
        available_actions = my_env.available_actions()
        # --- Step 1: play winning move, if possible ---
        for action in available_actions:
            nstate = after_action_state(state, action)
            gstatus = check_game_status(nstate[0])
            if gstatus > 0:
                if tomark(gstatus) == self.mark:
                    return action

        # --- Step 2: block opponent from winning ---
        # imagine the opponent was playing
        rev_state = (state[0], next_mark(state[1]))
        for action in available_actions:
            nstate = after_action_state(rev_state, action)
            gstatus = check_game_status(nstate[0])
            if gstatus > 0:
                # if they can make a winning move, play that
                if tomark(gstatus) == self.opponent_mark:
                    return action

        return random.choice(available_actions)


# =============== TicTacPro ===============


T = TypeVar("T")


class Node(Generic[T]):
    """A basic node for building search tree"""

    def __init__(self, state, action, parent: Optional["Node[T]"] = None):
        global NODE_ID_COUNTER

        self.id = NODE_ID_COUNTER
        NODE_ID_COUNTER += 1

        self.state = state
        self.action = action

        self.visits = 0  # n
        self.value = 0  # q

        self.parent = parent
        self.children: List["Node[T]"] = []

        if parent:
            self.update_parent_callback()

    @property
    def unvisited(self):
        return self.visits == 0

    def update_parent_callback(self):
        """Create two-way link"""
        self.parent.children.append(self)

    def backpropogate_score(self, inc_visits, inc_value):
        self.visits += inc_visits
        self.value += inc_value
        if self.parent:
            self.parent.backpropogate_score(inc_visits, inc_value)

    def __repr__(self):
        return f"{self.id}: {self.value}/{self.visits}"


class TicTacPro(BaseAgent):
    """
    Monte Carlo Tree Search Based
    """

    name = "TicTacPro"

    def __init__(self, mark: Mark, n_iter=10000, c=50):
        """
        Parameters
        ----------
        n_iter: int
                The number of iterations of the
                selection/expansion/simulation/backpropogation
                process to carry out before making a move.
        c: int
                Confidence constant used in UCT formula.
        """
        self.mark = mark
        self.n_iter = n_iter
        self.c = c

    def act(self, starting_state, env: TicTacToeEnv):
        """
        Given a current board state & TicTacToe environment,
        build up tree then select best seen move.
        """

        global NODE_ID_COUNTER
        NODE_ID_COUNTER = 0

        root_node: Node = Node(starting_state, action=None, parent=None)

        for _ in range(self.n_iter):
            my_env = deepcopy(env)
            # --- selection ---
            node, my_env = self.select(root_node, my_env)
            # --- expansion ---
            node, my_env = self.expand(node, my_env)
            # --- simulation ----
            reward = self.simulate(node, my_env)
            # --- backpropogation ---
            node.backpropogate_score(inc_visits=1, inc_value=reward)

        # As action, select child with highest number of visits
        best_child = max(root_node.children, key=lambda x: x.visits)

        return best_child.action

    def select(self, node: Node, my_env: TicTacToeEnv):
        """
        MCTS: Selection stage.
            - If node has any unvisted children, select one
            - If all children visited, choose best by UCB score
              and advance environment
            - If no children, return itself.
        """
        while node.children:
            # if any unexplored children, pick one
            univisted_children = [c for c in node.children if c.unvisited]
            if univisted_children:
                node = univisted_children[0]
            # otherwise, choose best child according to ucb
            else:
                node = max(node.children, key=self.ucb_score)
            # step copied environment forward
            my_env.step(node.action)
        return node, my_env

    def expand(self, node: Node, my_env: TicTacToeEnv):
        """
        MCTS: Expansion stage.
          - If additional moves are possible from given node
            child nodes will be created, one selected, and env advanced.
          - If not, same node and env will be returned.
        """
        # If this is a terminal state, don't try to expand
        if my_env.done:
            return node, my_env

        # Add a child node for each possible action
        for action in my_env.available_actions():
            nstate = after_action_state(node.state, action)
            Node(nstate, action, parent=node)

        # If node has children after expansion, select one
        if node.children:
            node = random.choice(node.children)
            my_env.step(node.action)

        return node, my_env

    def simulate(self, node: Node, my_env: TicTacToeEnv) -> float:
        """
        MCTS: Simulation stage.
            - Randomly play out remainder of moved and report reward

        Won reward=1, Tie reward=0.5, Lost reward=0
        """
        state = node.state
        while not my_env.done:
            action = random.choice(my_env.available_actions())
            state, _, _, _ = my_env.step(action)
        return self.compute_reward(state)

    def ucb_score(self, node: Node):
        """Given a node, return UCB score. Uses self.c as confidence constant."""
        if not node.parent:
            raise RuntimeError("UCB score called for node with no parent.")

        return (node.value / node.visits) + self.c * math.sqrt(
            (math.log(node.parent.visits)) / node.visits
        )

    def compute_reward(self, state):
        """Given a terminal state, return reward"""
        gstatus = check_game_status(state[0])
        if gstatus == -1:
            raise RuntimeError(
                "Error! Compute reward called when game was not finished."
            )
        # tie
        elif gstatus == 0:
            return 0.5
        # won
        elif gstatus == tocode(self.mark):
            return 1
        # lost
        else:
            return 0
