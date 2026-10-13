import chess 

from base import MinMax as minmax, function as heuristic

import torch
from ANN import ANN
import numpy as np 


model = ANN()
model.load_state_dict(torch.load("chess_ann.pt"))
model.eval()    

color_to_turn = {
    chess.WHITE: 1,
    chess.BLACK: -1
}
piece_order = [
    chess.Piece(chess.PAWN,   chess.WHITE), chess.Piece(chess.KNIGHT, chess.WHITE),
    chess.Piece(chess.BISHOP, chess.WHITE), chess.Piece(chess.ROOK,   chess.WHITE),
    chess.Piece(chess.QUEEN,  chess.WHITE), chess.Piece(chess.KING,   chess.WHITE),
    chess.Piece(chess.PAWN,   chess.BLACK), chess.Piece(chess.KNIGHT, chess.BLACK),
    chess.Piece(chess.BISHOP, chess.BLACK), chess.Piece(chess.ROOK,   chess.BLACK),
    chess.Piece(chess.QUEEN,  chess.BLACK), chess.Piece(chess.KING,   chess.BLACK),
]
piece_to_onehot = {piece: np.eye(12, dtype=np.float32)[index] for index, piece in enumerate(piece_order)}

color_to_turn = {
    chess.WHITE: 1,
    chess.BLACK: -1
}

def board_to_numerical_rep(board):
        state_tensor = np.zeros(769, dtype=np.float32)
        for square, piece in board.piece_map().items():
            state_tensor[square * 12 : (square + 1) * 12] = piece_to_onehot[piece]
        state_tensor[768] = color_to_turn[board.turn]
        return state_tensor

def deep_network(board_state, color):
    X = board_to_numerical_rep(board_state)

    X = torch.from_numpy(X).unsqueeze(0)

    with torch.no_grad():
        raw_score = model(X).item()

    return raw_score if color == chess.WHITE else -raw_score


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

tests = {
    "black down a queen": "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "start":              chess.STARTING_FEN,
    "white down a queen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1",
}
for name, fen in tests.items():
    print(f"{name:20s}", deep_network(chess.Board(fen), chess.WHITE))