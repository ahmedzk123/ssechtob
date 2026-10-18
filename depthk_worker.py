import math

import chess
import numpy as np
import torch

from base import MinMax, function

"""
helpers for learndepthK_nn.ipynb (v2)
- lives in a .py cuz windows multiprocessing cant pickle stuff defined inside a notebook
- encode: board flipped to side to move, 768 pieces + 4 castling = 772
- nets get loaded once per worker (cached) and run in numpy, way faster than a torch call per board
- quiescence: at the leaf we keep searching captures so we dont eval mid trade
"""

# slot per square: 0-5 = my P N B R Q K, 6-11 = opp P N B R Q K
piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]

def encode(board, dtype=np.float32):
    b = board if board.turn == chess.WHITE else board.mirror()   # now "white" = side to move
    state_tensor = np.zeros(772, dtype=dtype)
    for square, piece in b.piece_map().items():
        slot = piece_types.index(piece.piece_type) + (0 if piece.color == chess.WHITE else 6)
        state_tensor[square * 12 + slot] = 1
    state_tensor[768] = b.has_kingside_castling_rights(chess.WHITE)
    state_tensor[769] = b.has_queenside_castling_rights(chess.WHITE)
    state_tensor[770] = b.has_kingside_castling_rights(chess.BLACK)
    state_tensor[771] = b.has_queenside_castling_rights(chess.BLACK)
    return state_tensor


# numpy version of ANN2.forward

class NumpyNet():
    def __init__(self, model_path):
        #sd = torch.load(model_path)
        sd = torch.load(model_path, map_location="cpu")
        self.layers = [(sd[f"W{i}.weight"].numpy(), sd[f"W{i}.bias"].numpy()) for i in range(5)]
        self.a = sd["a"].item()
        self.piece_values = sd["piece_values"].numpy()

    def __call__(self, x):
        material = x @ self.piece_values
        h = x
        for W, b in self.layers[:-1]:
            h = W @ h + b
            h = np.where(h > 0, h, 0.01 * h)
        W, b = self.layers[-1]
        return math.tanh((W @ h + b).item() + self.a * material / 4)

_nets = {}

def get_eval(model_path):
    # None -> material, else nn eval from colors pov (nn outputs side to move pov)
    if model_path is None: return function
    if model_path not in _nets:
        torch.set_num_threads(1)
        _nets[model_path] = NumpyNet(model_path)
    net = _nets[model_path]
    def nn_eval(board_state, color):
        s = net(encode(board_state))
        return s if board_state.turn == color else -s
    return nn_eval


# quiescence wrapped as an eval fn, so base.MinMax stays untouched

piece_value = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}

def mvv_lva(board, move):
    victim = board.piece_at(move.to_square)
    victim_value = piece_value[victim.piece_type] if victim else 1   # en passant
    return 10 * victim_value - piece_value[board.piece_at(move.from_square).piece_type]

def with_quiescence(eval_fn, max_captures=4):
    def q(board, alpha, beta, depth):
        stand_pat = eval_fn(board, board.turn)
        if stand_pat >= beta or depth == 0: return stand_pat
        alpha = max(alpha, stand_pat)
        captures = sorted((m for m in board.legal_moves if board.is_capture(m)),
                          key=lambda m: mvv_lva(board, m), reverse=True)
        for move in captures:
            board.push(move)
            score = -q(board, -beta, -alpha, depth - 1)
            board.pop()
            if score >= beta: return score
            alpha = max(alpha, score)
        return alpha

    def q_eval(board_state, color):
        s = q(board_state, -float('inf'), float('inf'), max_captures)
        return s if board_state.turn == color else -s
    return q_eval


def to_label(score, is_material):
    # side to move pov, in [-1, 1]
    if math.isinf(score): return 1.0 if score > 0 else -1.0
    return math.tanh(score / 4) if is_material else score

def label_fen(args):
    fen, d, max_captures, model_path = args
    board = chess.Board(fen)
    engine = MinMax(with_quiescence(get_eval(model_path), max_captures), board.turn)
    score = engine.minmax_depth_k(board, d, True, -float('inf'), float('inf'))
    return to_label(score, model_path is None)


# games: spec = None (material) or a model path, both sides use plain base.MinMax

def play_game(args):
    white_spec, black_spec, depth, seed, max_plies = args
    rng = np.random.default_rng(seed)
    board = chess.Board()
    for _ in range(int(rng.integers(4, 9))):
        moves = list(board.legal_moves)
        board.push(moves[rng.integers(len(moves))])
    engines = {chess.WHITE: MinMax(get_eval(white_spec), chess.WHITE),
               chess.BLACK: MinMax(get_eval(black_spec), chess.BLACK)}
    while not board.is_game_over() and board.ply() < max_plies:
        board.push(engines[board.turn].find_best_move(board.copy(), depth, True))

    if board.is_game_over():
        return board.result(), False
    # hit the ply cap: whoever is up material wins
    m = function(board, chess.WHITE)
    return ("1-0" if m > 0 else "0-1" if m < 0 else "1/2-1/2"), True
