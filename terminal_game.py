import chess 

from base import MinMax as minmax, function as heuristic

from depthk_worker import get_eval

# v2 depthk nn (flipped 772 input ANN2), runs in numpy
deep_network = get_eval("chess_depthk2_r3.pt")


# game below

def print_board(board, perspective=chess.WHITE):
    rows = str(board).split("\n")          # 8 lines, rank 8 first
    files = "a b c d e f g h"

    if perspective == chess.BLACK:         # flip so the player's pieces are at the bottom
        rows = [r[::-1] for r in reversed(rows)]
        files = "h g f e d c b a"
        ranks = range(1, 9)
    else:
        ranks = range(8, 0, -1)

    print()
    for rank, row in zip(ranks, rows):
        print(rank, row)
    print("  " + files)
    print()

"""

board = chess.Board()  # rough example, verify the FEN yourself

choice = ""
while choice not in ("W", "B"):
    choice = input("W or B? ").strip().upper()

player_color = chess.WHITE if choice == "W" else chess.BLACK

#bot_engine = minmax(heuristic, not player_color)
player_engine = minmax(heuristic, player_color)
ann_bot_engine = minmax(deep_network, not player_color)


print_board(board, player_color)

while not board.is_game_over():

    if board.turn == player_color:
        try:
            #player_move = input("Your move: ").strip()
            #board.push_uci(player_move)

            player_move = player_engine.find_best_move(board.copy(), 5, True)
            board.push(player_move)
            print_board(board, player_color)

        except ValueError as error:
            print("Illegal move: ", error)
    else:
        try:
            #bot_move = bot_engine.find_best_move(board.copy(), 5, True)
            ann_move = ann_bot_engine.find_best_move(board.copy(), 5, True)
            board.push(ann_move)
            print("bot played", ann_move.uci())
            print_board(board, player_color)
            

        except ValueError as error:
            print("Illegal move: ", error)

outcome = board.outcome()         

"""

"""
tests = {
    "black down a queen": "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "start":              chess.STARTING_FEN,
    "white down a queen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1",
}
for name, fen in tests.items():
    print(f"{name:20s}", deep_network(chess.Board(fen), chess.WHITE))
"""

# material bot vs depthk nn bot, 5 games, swap colours, few random opening moves so games differ

import random
random.seed(0)

depth = 3
num_games = 5
score = {"nn": 0, "material": 0, "draw": 0}

for g in range(num_games):
    nn_color = chess.WHITE if g % 2 == 0 else chess.BLACK
    engines = {nn_color: minmax(deep_network, nn_color), not nn_color: minmax(heuristic, not nn_color)}

    board = chess.Board()
    for _ in range(4):
        board.push(random.choice(list(board.legal_moves)))

    while not board.is_game_over() and board.ply() < 200:
        board.push(engines[board.turn].find_best_move(board.copy(), depth, True))

    outcome = board.outcome()
    if outcome is None or outcome.winner is None: winner = "draw"
    else: winner = "nn" if outcome.winner == nn_color else "material"
    score[winner] += 1
    print(f"game {g+1}: nn {'white' if nn_color == chess.WHITE else 'black'} -> {board.result()} ({winner})")

print(score)