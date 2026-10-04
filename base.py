import chess 

"""

will ipdate k==0 case to return a NN that takes in board and ooutpiuts best score 
- kinda like rollout idea but willl hopefilly train NN to find better soiutojns
- needa implement alpha beta pruning 


root node will always be maximinzing  (each case starts fresh)
- so we only care about  max best mvoe


-- soeting matter alot fro pruning, maybe have NN to sort before pruning
-- for now well sort by capture moves

"""

class MinMax():
    def __init__(self, eval, color):
        self.eval = eval
        self.color = color

    def ordered_moves(self, board):
        return sorted(board.legal_moves, key=board.is_capture, reverse=True)


    def find_best_move(self, board, k, max_node):
        
        alpha = -float('inf')
        beta = float('inf')
        best_move = None

        
        for move in self.ordered_moves(board):
            board.push(move)
            score = self.minmax_depth_k(board, k-1, not max_node, alpha, beta)
            board.pop()  
        
            if score > alpha:
                alpha = score
                best_move = move

            elif best_move == None:
                best_move = move
        
        return best_move

    def minmax_depth_k(self, board, k, max_node, alpha, beta):

        # alpha: largest value a max node can have at depth d
        # beta: smallest value a min node can have at depth d

        if max_node:
            if board.is_checkmate(): return -float('inf')
            if board.is_game_over(): return 0
            if k == 0: return self.eval(board, self.color)  
            
            best_score = -float('inf')
            for move in self.ordered_moves(board):
                board.push(move)
                move_k_score = self.minmax_depth_k(board, k-1, not max_node, alpha, beta)
                board.pop()
                if move_k_score > best_score: 
                    best_score = move_k_score
                    alpha = max(best_score, alpha)
                if alpha >= beta:
                    return best_score
            return best_score
            
                
        elif not max_node:
            if board.is_checkmate(): return float('inf')
            if board.is_game_over(): return 0
            if k == 0: return self.eval(board, self.color)
            
            opp_score = float('inf')
            for move in self.ordered_moves(board):
                board.push(move)
                move_k_score = self.minmax_depth_k(board, k-1, not max_node, alpha, beta)
                board.pop()
                if move_k_score < opp_score: 
                    opp_score = move_k_score
                    beta = min(opp_score, beta)
                if beta <= alpha:
                    return opp_score
            return opp_score


def function(board_state, color):
    values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
              chess.ROOK: 5, chess.QUEEN: 9}
    score = 0
    for piece_type, value in values.items():
        score += len(board_state.pieces(piece_type, color)) * value
        score -= len(board_state.pieces(piece_type, not color)) * value
    return score

"""
board = chess.Board("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1")  # rough example, verify the FEN yourself
board_2 = chess.Board("4k3/8/8/3q4/8/8/8/3RK3 w - - 0 1")
board_3 = board_2.mirror()
board_4 = chess.Board("7k/8/8/8/8/8/R7/1R4K1 w - - 0 1")
board_5 = chess.Board("7k/8/6K1/8/8/8/8/5Q2 w - - 0 1")
board_6 = chess.Board("8/P7/8/8/8/8/k7/4K3 w - - 0 1")

# board, depth (k) , expeected move

tests = [
    (board,   1, "a1a8"),
    (board_2, 2, "d1d5"),
    (board_3, 2, "d8d4"),
    (board_4, 3, "a2a7"),   # b1b7 is also a valid mate in 2
    (board_5, 2, "f1f8"),
    (board_6, 2, "a7a8q"),
]

for i, (b, k, expected) in enumerate(tests):
    engine = MinMax(function, b.turn)   # color = side to move
    move = engine.find_best_move(b, k, max_node=True)
    print(i, move.uci(), "OK" if move.uci() == expected else f"FAIL (expected {expected})")
"""

import time, random

def plain_minimax(board, k, max_node, eval_fn, color):
    if board.is_checkmate(): return -float('inf') if max_node else float('inf')
    if board.is_game_over(): return 0
    if k == 0: return eval_fn(board, color)
    scores = []
    for move in list(board.legal_moves):
        board.push(move)
        scores.append(plain_minimax(board, k-1, not max_node, eval_fn, color))
        board.pop()
    return max(scores) if max_node else min(scores)

def plain_root(board, k, eval_fn, color):
    """Returns (best_score, {move: score}) using no pruning."""
    scores = {}
    for move in list(board.legal_moves):
        board.push(move)
        scores[move] = plain_minimax(board, k-1, False, eval_fn, color)
        board.pop()
    return max(scores.values()), scores

def random_position(max_plies=40):
    b = chess.Board()
    for _ in range(random.randint(4, max_plies)):
        if b.is_game_over(): break
        b.push(random.choice(list(b.legal_moves)))
    return b

def compare(n=30, k=3, seed=1):
    random.seed(seed)
    t_plain = t_pruned = 0
    mismatches = tested = 0
    for _ in range(n):
        b = random_position()
        if b.is_game_over(): continue
        tested += 1
        engine = MinMax(function, b.turn)

        t0 = time.time()
        best_score, scores = plain_root(b, k, function, b.turn)
        t_plain += time.time() - t0

        t0 = time.time()
        move = engine.find_best_move(b, k, max_node=True)
        t_pruned += time.time() - t0

        # the pruned engine's move must have the same score as plain minimax's best
        if scores[move] != best_score:
            mismatches += 1
            print("MISMATCH:", b.fen(), move, scores[move], "vs best", best_score)

    print(f"positions: {tested}, mismatches: {mismatches}")
    print(f"plain:  {t_plain:.2f}s")
    print(f"pruned: {t_pruned:.2f}s  ({t_plain / t_pruned:.1f}x faster)")

# compare(n=30, k=3)



