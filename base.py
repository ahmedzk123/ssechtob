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


class Quiescence():
    def __init__(self, eval_fn, color, max_captures=4):
        self.eval_fn = eval_fn
        self.max_captures = max_captures
        self.color = color

    def ordered_moves(self, board):
            return sorted(board.legal_moves, key=board.is_capture, reverse=True)

    def find_best_move(self, board, k, max_node):
        alpha = -float('inf')
        beta = float('inf')
        best_move = None

        for move in self.ordered_moves(board):
            board.push(move)
            score = self.quiescence(board, not max_node, alpha, beta, k-1, 0)
            board.pop()

            if score > alpha:
                alpha = score
                best_move = move

            elif best_move == None:
                best_move = move

        return best_move

    def quiescence(self, board, max_node, alpha, beta, tree_depth, capture_depth):

        if max_node:
            if board.is_checkmate(): return -float('inf')
            if board.is_game_over(): return 0

            if tree_depth > 0: 
                for move in self.ordered_moves(board):
                    board.push(move)
                    score = self.quiescence(board, not max_node, alpha, beta, tree_depth - 1, capture_depth)
                    board.pop()
                    if score > alpha:
                        alpha = score
                    if alpha >= beta:
                        return alpha

            # quesisicne shines
            if tree_depth == 0:
                if board.is_check():
                    for move in self.ordered_moves(board):
                        board.push(move)
                        score = self.quiescence(board, not max_node, alpha, beta, 0, capture_depth + 1)
                        board.pop()
                        if score > alpha:
                            alpha = score
                        if alpha >= beta:
                            return alpha
                else:
                    # stand pat, at max_captures we just eval (no more captures)
                    board_state_score = self.eval_fn(board, self.color)
                    if board_state_score > alpha:
                        alpha = board_state_score
                    if capture_depth < self.max_captures:
                        for move in self.ordered_moves(board):
                            if not board.is_capture(move) and not move.promotion: continue
                            board.push(move)
                            score = self.quiescence(board, not max_node, alpha, beta, 0, capture_depth + 1)
                            board.pop()
                            if score > alpha:
                                alpha = score
                            if alpha >= beta:
                                return alpha
                return alpha
            return alpha
        
                    

        else: # not max node ~ min node
            if board.is_checkmate(): return float('inf')
            if board.is_game_over(): return 0

            if tree_depth > 0: 
                for move in self.ordered_moves(board):
                    board.push(move)
                    score = self.quiescence(board, not max_node, alpha, beta, tree_depth - 1, capture_depth)
                    board.pop()
                    if score < beta:
                        beta = score
                    if beta <= alpha:
                        return beta

            if tree_depth == 0:
                if board.is_check():
                    for move in self.ordered_moves(board):
                        board.push(move)
                        score = self.quiescence(board, not max_node, alpha, beta, 0, capture_depth + 1)
                        board.pop()
                        if score < beta:
                            beta = score
                        if beta <= alpha:
                                return beta
                else:
                    board_state_score = self.eval_fn(board, self.color)
                    beta = min(beta, board_state_score)
                    if capture_depth < self.max_captures:
                        for move in self.ordered_moves(board):
                            if not board.is_capture(move) and not move.promotion: continue
                            board.push(move)
                            score = self.quiescence(board, not max_node, alpha, beta, 0, capture_depth + 1)
                            board.pop()
                            if score < beta:
                                beta = score
                            if beta <= alpha:
                                return beta
                return beta
            return beta
            
        
        

