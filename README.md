# Monte Carlo Tic-Tac-Toe

<img src="https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fupload.wikimedia.org%2Fwikipedia%2Fcommons%2Fthumb%2F3%2F32%2FTic_tac_toe.svg%2F1200px-Tic_tac_toe.svg.png&f=1&nofb=1" width="150"/>

Here is an implementation of an AI agent, `TicTacPro` which plays tic-tac-toe using Monte Carlo tree search. The underlying game environment is provided by [gym-tictactoe](https://github.com/haje01/gym-tictactoe), and the agents are modeled after their example code.

## Usage

```sh
# install requirements
pip install -r requirements.txt

# simulate some games
python run.py
```

## Code Layout

- The Monte Carlo Tree Search is implemented inside `agents.py`, for the AI agent `TicTacPro`.

- Like the other, simpler agents, `TicTacPro` has an `act` method which takes the current state & environment, and returns the move it wants to make. Each time act is called, the agent builds up a new tree.

- In `run.py`, this agent can be made to play against its historic rival, `TicTacJoe`.
