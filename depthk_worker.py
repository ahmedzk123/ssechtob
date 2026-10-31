import math
import random

import chess
import torch

from ANN import ANN2
from base import MinMax, Quiescence, function

"""
helpers for learndepthK_nn.ipynb (v2)
- lives in a .py cuz windows multiprocessing cant pickle stuff defined inside a notebook
- encode: board flipped to side to move, 768 pieces + 4 castling = 772
- models get loaded once per worker (cached)
- search_method: base.Quiescence (keeps searching captures at the leaf so we dont eval mid trade) or plain base.MinMax
"""

# slot per square: 0-5 = my P N B R Q K, 6-11 = opp P N B R Q K
piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]


# map a chess.Board to a 772-dim tensor (side to move is always white)
def encode(board, dtype=torch.float32):
    b = board if board.turn == chess.WHITE else board.mirror()  
    state_tensor = torch.zeros(772, dtype=dtype)
    for square, piece in b.piece_map().items():
        slot = piece_types.index(piece.piece_type) + (0 if piece.color == chess.WHITE else 6)
        state_tensor[square * 12 + slot] = 1
    state_tensor[768] = b.has_kingside_castling_rights(chess.WHITE)
    state_tensor[769] = b.has_queenside_castling_rights(chess.WHITE)
    state_tensor[770] = b.has_kingside_castling_rights(chess.BLACK)
    state_tensor[771] = b.has_queenside_castling_rights(chess.BLACK)
    return state_tensor


# models/engine stuff

_models = {}
def get_eval(model_path):
    if model_path is None: return function
    if model_path not in _models:
        torch.set_num_threads(1)
        model = ANN2()
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()
        _models[model_path] = model
    model = _models[model_path]
    def nn_eval(board_state, color):
        with torch.no_grad():
            s = model(encode(board_state)).item()
        return s if board_state.turn == color else -s
    return nn_eval


def make_engine(search_method, spec, color, max_captures):
    eval_fn = get_eval(spec)
    if search_method is Quiescence:
        return Quiescence(eval_fn, color, max_captures)
    return MinMax(eval_fn, color)


def search_score(fen, depth, max_captures, model_path, search_method=Quiescence):
    board = chess.Board(fen)
    engine = make_engine(search_method, model_path, board.turn, max_captures)
    if search_method is Quiescence:
        return engine.quiescence(board, True, -float('inf'), float('inf'), depth, 0)
    return engine.minmax_depth_k(board, depth, True, -float('inf'), float('inf'))


# labelleing

def to_label(score, is_material):
    # side to move pov, in [-1, 1]
    if math.isinf(score): return 1.0 if score > 0 else -1.0
    return math.tanh(score / 4) if is_material else score

def label_fen(args):
    model_path = args[3]
    return to_label(search_score(*args), model_path is None)


# games

def random_opening(seed):
    rng = random.Random(seed)
    board = chess.Board()
    for _ in range(rng.randint(4, 8)):
        board.push(rng.choice(list(board.legal_moves)))
    return board

def run_game(white_spec, black_spec, depth, seed, max_plies, search_method=Quiescence, max_captures=4):
    board = random_opening(seed)
    white_engine = make_engine(search_method, white_spec, chess.WHITE, max_captures)
    black_engine = make_engine(search_method, black_spec, chess.BLACK, max_captures)

    while not board.is_game_over() and board.ply() < max_plies:
        engine = white_engine if board.turn == chess.WHITE else black_engine
        board.push(engine.find_best_move(board.copy(), depth, True))

    if board.is_game_over():
        return board.result(), False
    # hit the ply cap: whoever is up material wins
    material = function(board, chess.WHITE)
    if material > 0: return "1-0", True
    if material < 0: return "0-1", True
    return "1/2-1/2", True

def play_game(args):
    # Pool.map passes one tuple: (white, black, depth, seed, max_plies[, search_method, max_captures])
    return run_game(*args)
